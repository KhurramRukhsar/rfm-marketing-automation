import streamlit as st
import pandas as pd
import json
from io import BytesIO

from config import setup_logging
from examples_loader import load_segment_examples
from db import connect_db, get_or_create_shortlink
from recommendations import load_product_data, hybrid_recommendations
from templates import generate_segment_templates
from utils import clean_phone, format_products_for_sms
from sms import send_sms_batch

logger = setup_logging()
logger.info("Application started")

segment_examples = load_segment_examples("urdu_sms_examples_by_segment.txt")

try:
    conn, cur = connect_db()
except Exception as e:
    logger.error("Failed to connect to PostgreSQL: %s", e)
    st.error(f"Cannot connect to PostgreSQL for link tracking: {e}\nSet PG_* env vars correctly or change the defaults.")
    st.stop()

try:
    load_product_data("your_cleaned_product_data.csv")
except FileNotFoundError:
    logger.error("your_cleaned_product_data.csv not found")
    st.error("❌ 'your_cleaned_product_data.csv' not found in the working directory.")
    st.stop()
except Exception as e:
    logger.error("Error loading product data: %s", e)
    st.error(f"❌ Error loading product data: {e}")
    st.stop()

st.title("BaKhabar SMS Recommendation System (Segment-Based)")
st.markdown("Upload your customer list, generate Urdu SMS templates, personalize them, and send SMS with tracking.")

uploaded_file = st.file_uploader("Upload Customer File (Excel)", type=["xlsx"])

if uploaded_file:
    logger.info("User uploaded file: %s", uploaded_file.name)

    users_df = pd.read_excel(
        uploaded_file,
        engine='openpyxl',
        dtype=str,
        keep_default_na=False,
        na_filter=False
    )

    for col in users_df.columns:
        users_df[col] = users_df[col].astype(str).str.split('.').str[0].str.strip()

    users_df.replace('nan', '', inplace=True)

    def get_col(df, names):
        cols = {c.lower().strip(): c for c in df.columns}
        for name in names:
            key = name.lower().strip()
            if key in cols:
                return cols[key]
        return None

    contact_col = get_col(users_df, ["Contact", "contact", "Phone", "phone", "MSISDN", "msisdn", "Mobile", "mobile"])
    segment_col = get_col(users_df, ["Segment", "segment"])
    recency_col = get_col(users_df, ["recency", "Recency"])
    freq_col = get_col(users_df, ["Total Orders", "total orders", "Frequency", "frequency"])
    money_col = get_col(users_df, ["Money Spent", "money spent", "Monetary", "monetary"])
    urdu_name_col = get_col(users_df, ["urdu_name", "Urdu Name", "urdu name", "نام اردو"])
    custid_col = get_col(users_df, ["customer_id", "Customer_ID", "Customer Id", "customer id"])

    if not contact_col:
        st.error("Error: Could not find a phone column. Please use one of: Contact, Phone, MSISDN, etc.")
        st.stop()
    if not segment_col:
        st.error("Error: Could not find 'Segment' column.")
        st.stop()
    if not custid_col:
        st.error("Error: Could not find 'customer_id' column.")
        st.stop()
    if not urdu_name_col:
        st.warning("⚠️ Could not find 'urdu_name' column. Will use 'Customer' as fallback name.")

    if custid_col != "customer_id":
        users_df["customer_id"] = users_df[custid_col].astype(str).str.strip()
    else:
        users_df["customer_id"] = users_df["customer_id"].astype(str).str.strip()

    try:
        strategy_df = pd.read_excel("Strategy.xlsx", engine='openpyxl')
        logger.info("Loaded Strategy.xlsx successfully")
    except FileNotFoundError:
        logger.error("Strategy.xlsx not found")
        st.error("Strategy.xlsx' not found in the working directory.")
        st.stop()

    strategy_map = strategy_df.set_index("Segment").to_dict(orient="index")

    st.subheader("Preview of Uploaded Data")
    st.dataframe(users_df.head())

    total_rows_to_process = 0
    for _, row in users_df.iterrows():
        if pd.isna(row.get(segment_col, None)):
            break
        total_rows_to_process += 1
    st.write(f"Total rows to process: {total_rows_to_process}")

    if st.button("📋 Fetch Purchase History"):
        st.subheader("🛒 Customer IDs and Hybrid Recommendations")
        customer_data = []
        for _, row in users_df.iterrows():
            customer_id = row.get('customer_id')
            if customer_id == "Total":
                break
            customer_id = str(customer_id).strip()
            recs = hybrid_recommendations(customer_id)
            recs_json = json.dumps([], indent=2) if recs.empty else json.dumps(
                recs[['product_id', 'pure_names', 'true_vendor', 'price', 'source']].to_dict(orient='records'), indent=2)
            customer_data.append({
                'customer_id': customer_id,
                'Recommendations': recs_json
            })

        customer_data_df = pd.DataFrame(customer_data)
        users_df['customer_id'] = users_df['customer_id'].astype(str).str.strip()
        customer_data_df['customer_id'] = customer_data_df['customer_id'].astype(str).str.strip()
        users_df = users_df.merge(customer_data_df[['customer_id', 'Recommendations']], on='customer_id', how='left')
        users_df['Recommendations'] = users_df['Recommendations'].fillna(json.dumps([]))
        st.session_state['users_df'] = users_df.copy()
        st.dataframe(users_df, use_container_width=True)

        all_recommendations = {row['customer_id']: json.loads(row['Recommendations']) for _, row in users_df.iterrows()}
        recommendations_json = json.dumps(all_recommendations, indent=2)
        json_bytes = BytesIO(recommendations_json.encode('utf-8'))

    if st.button("🚀 Generate Content (Segment-Based)"):
        if 'users_df' not in st.session_state:
            st.error("❌ No user data found in session state. Please run 'Fetch Purchase History' first.")
            st.stop()

        users_df = st.session_state['users_df'].copy()

        valid_users_df = users_df[~pd.isna(users_df[segment_col])].copy()
        if valid_users_df.empty:
            st.error("❌ No valid segments found in the data.")
            st.stop()

        unique_segments = valid_users_df[segment_col].unique()

        with st.spinner("⚙️ Generating message templates for each segment..."):
            segment_templates = generate_segment_templates(valid_users_df, segment_col, strategy_map, segment_examples)

        if not segment_templates:
            st.error("❌ Failed to generate any message templates.")
            st.stop()

        st.session_state["segment_templates"] = segment_templates.copy()
        st.session_state["valid_users_df"] = valid_users_df.copy()
        st.session_state["segment_col"] = segment_col
        st.session_state["strategy_map"] = strategy_map

    if "segment_templates" in st.session_state and "valid_users_df" in st.session_state:
        segment_templates = st.session_state["segment_templates"]
        valid_users_df = st.session_state["valid_users_df"]
        segment_col = st.session_state["segment_col"]
        strategy_map = st.session_state["strategy_map"]

        if st.session_state.get("regen_segment"):
            regen_segment = st.session_state["regen_segment"]
            filtered_df = valid_users_df[valid_users_df[segment_col] == regen_segment]
            with st.spinner("Regenerating template..."):
                new_template = generate_segment_templates(filtered_df, segment_col, strategy_map, segment_examples)
            if regen_segment in new_template:
                segment_templates[regen_segment] = new_template[regen_segment]
                st.session_state["segment_templates"][regen_segment] = new_template[regen_segment]
            st.session_state["regen_segment"] = None

        st.subheader("📋 Generated Message Templates")
        for segment in segment_templates:
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"**Segment:** {segment}")
                st.code(segment_templates[segment], language="text")
            with cols[1]:
                if st.button("Regenerate", key=f"regen_{segment}"):
                    st.session_state["regen_segment"] = segment
                    st.rerun()

        st.divider()
        approval = st.radio(
            "Do you approve the templates and want to generate personalized messages?",
            ["No", "Yes"],
            horizontal=True
        )

        if approval == "Yes" and st.button("Generate Personalized Messages"):
            st.info("🔧 Personalizing messages for each user with their recommended products...")
            generated_messages = []
            short_codes = []
            short_links = []
            personalized_count = 0

            unique_segments = valid_users_df[segment_col].unique()

            progress_bar = st.progress(0)
            status_text = st.empty()

            for _, row in valid_users_df.iterrows():
                segment = row.get(segment_col)
                if not segment or segment not in segment_templates:
                    generated_messages.append("")
                    short_codes.append(None)
                    short_links.append("")
                    continue

                raw_contact = row.get(contact_col) or ""
                msisdn = clean_phone(raw_contact)
                if not msisdn:
                    generated_messages.append("")
                    short_codes.append(None)
                    short_links.append("")
                    continue

                original_url = row.get("RecommendedProductLink", "https://shop.bkk.ag/")
                short_code, short_link = get_or_create_shortlink(cur, raw_contact, original_url=original_url)
                if not short_link:
                    short_link = "https://shop.bkk.ag/"
                    short_code = None

                short_codes.append(short_code)
                short_links.append(short_link)

                urdu_name = row.get(urdu_name_col, '')
                if pd.isna(urdu_name) or urdu_name == 'nan' or not str(urdu_name).strip():
                    customer_name = "کسٹمر"
                else:
                    customer_name = str(urdu_name).strip()

                recommendations_json = row.get('Recommendations', '[]')
                formatted_products = format_products_for_sms(recommendations_json)

                template = segment_templates[segment]
                personalized_message = template.replace("{name}", customer_name).replace("{products}", formatted_products)

                if not formatted_products.strip():
                    msg_lines = [ln for ln in personalized_message.splitlines() if ln.strip()]
                    msg_lines = [ln[:-1] if ln.endswith(":") else ln for ln in msg_lines]
                    personalized_message = "\n".join(msg_lines)

                if short_link not in personalized_message:
                    personalized_message = f"{personalized_message.strip()}\n{short_link}"

                message_length = len(personalized_message.replace("\n", " ").strip())
                if message_length > 450:
                    base_message = personalized_message.replace(short_link, "").strip()
                    if len(base_message.replace("\n", " ")) > 350:
                        lines = base_message.split("\n")
                        if len(lines) > 4:
                            trimmed_message = "\n".join(lines[:3] + ["..."] + [lines[-1]])
                            personalized_message = f"{trimmed_message}\n{short_link}"

                generated_messages.append(personalized_message)
                personalized_count += 1

                progress_bar.progress(len(generated_messages) / len(valid_users_df))
                status_text.text(f"📝 Personalized {len(generated_messages)} of {len(valid_users_df)} messages")

            users_df["GeneratedContent"] = ""
            users_df["short_code"] = None
            users_df["short_link"] = ""

            generated_series = pd.Series(generated_messages, index=valid_users_df.index)
            short_code_series = pd.Series(short_codes, index=valid_users_df.index)
            short_link_series = pd.Series(short_links, index=valid_users_df.index)

            users_df.loc[generated_series.index, "GeneratedContent"] = generated_series
            users_df.loc[short_code_series.index, "short_code"] = short_code_series
            users_df.loc[short_link_series.index, "short_link"] = short_link_series
            st.session_state["final_df"] = users_df.copy()

            st.success(f"✅ Successfully personalized {personalized_count} messages!")
            st.write("📊 Statistics:")
            st.write(f"- Total users processed: {len(valid_users_df)}")
            st.write(f"- Unique segments: {len(unique_segments)}")
            st.write(f"- LLM API calls made: {len(segment_templates)} (instead of {len(valid_users_df)})")
            st.write(f"- Messages personalized: {personalized_count}")

            st.subheader("📱 Sample Personalized Messages")
            sample_count = min(3, len(generated_messages))
            for i in range(sample_count):
                if generated_messages[i]:
                    with st.expander(f"Sample Message {i+1}"):
                        st.text(generated_messages[i])
                        st.write(f"Length: {len(generated_messages[i].replace(chr(10), ' '))} characters")
                        st.write(f"Lines: {generated_messages[i].count(chr(10)) + 1}")

            st.subheader("📊 Final User Data with Personalized Messages")
            st.dataframe(users_df)

    st.subheader("📨 Send SMS to Users")

    if st.button("📤 Send SMS"):
        if 'final_df' not in st.session_state:
            st.warning("⚠️ Please generate content first before sending SMS.")
            st.stop()

        users_df = st.session_state['final_df']
        st.info("Sending SMS to valid phone numbers and saving results to database...")

        sms_results_df = send_sms_batch(users_df, contact_col, urdu_name_col, custid_col, total_rows_to_process, conn, cur)

        st.success("✅ SMS sending and database storage complete!")
        st.subheader("📋 SMS Results")
        st.dataframe(sms_results_df)
