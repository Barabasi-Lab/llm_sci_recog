"""Shared paths for the Section 2.1 pipeline.

Every script reads and writes under WORK (default: pipeline/work/, git-ignored).
Override with the environment variables below.

    DIMENSIONS_PANEL  CSV extract of the Dimensions author panel (one row per
                      researcher: author_id, names, modal_field_name, year,
                      cum_citations, cum_publication_count, Gender, career_stage,
                      affiliation (sub)regions). Dimensions data are licensed and
                      are not redistributed.
    PIPELINE_WORK     working directory for all intermediate files.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, "inputs")
WORK = os.environ.get("PIPELINE_WORK", os.path.join(HERE, "work"))
PANEL_CSV = os.environ.get("DIMENSIONS_PANEL",
                           os.path.join(WORK, "author_panel_complete.csv"))

SAMPLE_DIR = os.path.join(WORK, "samples")                      # 01 (input samples)
MULTI_DIR = os.path.join(WORK, "llm_recognition_multi")         # 01-03
REGION_DIR = os.path.join(WORK, "llm_recognition_region20k")    # 04 (input top-up sample)
TIMELINE_DIRS = {                                               # 05
    "gpt": os.path.join(WORK, "llm_recognition_timeline"),
    "claude": os.path.join(WORK, "llm_recognition_timeline_claude"),
    "gemini": os.path.join(WORK, "llm_recognition_timeline_gemini"),
}
