import logging


def load_segment_examples(path: str):
    logger = logging.getLogger(__name__)
    segment_examples = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            examples_text = f.read()
        current_segment = None
        current_example_lines = []
        for raw_line in examples_text.splitlines():
            line = raw_line.rstrip("\n")
            if line.startswith("SEGMENT:"):
                if current_segment and current_example_lines:
                    template = "\n".join(current_example_lines).strip()
                    if template:
                        segment_examples.setdefault(current_segment, []).append(template)
                current_segment = line.replace("SEGMENT:", "").strip()
                current_example_lines = []
                continue
            if line.strip().lower().startswith("example"):
                if current_segment and current_example_lines:
                    template = "\n".join(current_example_lines).strip()
                    if template:
                        segment_examples.setdefault(current_segment, []).append(template)
                current_example_lines = []
                continue
            if current_segment:
                current_example_lines.append(line)
        if current_segment and current_example_lines:
            template = "\n".join(current_example_lines).strip()
            if template:
                segment_examples.setdefault(current_segment, []).append(template)
        logger.info("Loaded %d segments of Urdu examples from %s", len(segment_examples), path)
    except FileNotFoundError:
        logger.warning("Examples file not found: %s", path)
        segment_examples = {}
    except Exception as e:
        logger.error("Failed to load examples file %s: %s", path, e)
        segment_examples = {}

    return segment_examples
