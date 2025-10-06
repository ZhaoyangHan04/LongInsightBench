# =============================================================================
# Common SYSTEM_PROMPT for all tasks
# =============================================================================
SYSTEM_PROMPT = """
You are a multimodal question generator specializing in video understanding.
Your task is to create high-quality multiple-choice questions (MCQs) for video understanding.
You are restricted to using ONLY the provided event list (visual_caption, audio_caption, timestamps).
Do not use external knowledge, hallucinated facts, or information not present in the events.
Each generated question must strictly follow the JSON schema below.
"""

# =============================================================================
# JSON Schema Definition
# =============================================================================
QUESTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "type": {"type": "string"}, # "single_choice_question" or "multiple_choice_question"
                    "question": {"type": "string"},
                    "options": {
                        "type": "object",
                        "properties": {
                            "A": {"type": "string"},
                            "B": {"type": "string"},
                            "C": {"type": "string"},
                            "D": {"type": "string"}
                        },
                        "required": ["A", "B", "C", "D"]
                    },
                    "required_event_ids": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "modalities_required": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "correct_answer": {
                        "type": "array",
                        "items": {"type": "string"} # Corresponds to "A", "B", "C", "D"
                    },
                    "related_video_id": {"type": "string"},
                    "gold_reasoning": {"type": "string"}
                },
                "required": [
                    "question_id", "type", "question", "options",
                    "required_event_ids", "modalities_required",
                    "correct_answer", "related_video_id", "gold_reasoning"
                ]
            }
        }
    },
    "required": ["questions"]
}


# =============================================================================
# USER_PROMPT Templates for different task types
# =============================================================================t
INTRA_EVENT_REASONING_USER_PROMPT = """
video_id: {video_id}
summary: {summary}
events: {events_str}

Task requirements:
    1) Generate N=2 **Intra-event Reasoning** questions. Each question MUST:
    - Focus on a single event, querying **causation or conclusions within its timestamp range** (e.g., "Why X happened / How Y was achieved / What Z signifies between [start_time] and [end_time]?").
    - Use **exactly 1 event_id**, referring only to its content and **timestamps**.
    - Require BOTH visual and audio evidence from that event; single modality is INSUFFICIENT.
    - Offer plausible options **strictly based on the specific event's details**.
    - Be "single_choice_question" or "multiple_choice_question".

    2) For answer options:
    - Provide exactly 4 options: A, B, C, D.
    - Distractors must be consistent with event details but incorrect.
    - For "single_choice_question": exactly one correct option.
    - For "multiple_choice_question": at least two correct options.

    3) For explanations:
    - Justify the correct answer(s) by synthesizing information **as if you were observing the video directly** and field the `"gold_reasoning"`.
    - Reasoning must detail inference steps, **explicitly referencing the single event_id and both visual + audio evidence.**
"""

MULTIMODAL_TEMPORAL_LOCALIZATION_USER_PROMPT = """
video_id: {video_id}
summary: {summary}
events: {events_str}

Task requirements:
    1) Generate N=2 **Multimodal Temporal Localization** questions. Each question MUST:
    - Focus on localizing a specific event, which is defined by the **simultaneous occurrence or strong correlation of a distinct visual action/cue AND associated audio information (e.g., speech content, specific sounds)**.
    - Ask for the exact time segment(s).
    - Use **exactly 1 event_id**. The question should provide enough detail from both visual and audio captions to uniquely identify the correct time segment(s).
    - Require BOTH visual and audio evidence from that event; single modality is INSUFFICIENT.
    - Offer plausible options **strictly based on the specific event's details**.
    - Be "single_choice_question" or "multiple_choice_question".

    2) For answer options:
    - Provide exactly 4 options: A, B, C, D. Each option's value **must be a timestamp string** in "[HH:MM:SS - HH:MM:SS]" format.
    - Distractor time segments must be plausible but incorrect for the queried event, ideally from other events or incorrect parts of the correct event.
    - For "single_choice_question": exactly one correct time segment option.
    - For "multiple_choice_question": at least two correct time segment options.

    3) For explanations:
    - Justify the correct answer(s) by synthesizing information **as if you were observing the video directly** and field the `"gold_reasoning"`.
    - Reasoning must detail inference steps, **explicitly referencing the required event_ids and both visual + audio evidence used to pinpoint the exact time segment.**
"""

AUDIO_VISUAL_ALIGNMENT_USER_PROMPT = """
video_id: {video_id}
summary: {summary}
events: {events_str}

Task requirements:
    1) Generate N=2 **Audio-Visual Alignment** questions. Each question MUST:
    - Focus on **identifying the corresponding visual characteristic/expression given an audio event, OR **identifying the corresponding audio event given a visual characteristic/** within a specific event.
    - Use **exactly 1 event_id**. The question should target an event's [start_time] and [end_time] where the specified audio and visual elements occur concurrently.
    - Require BOTH visual and audio evidence to correctly identify the aligning characteristic; single modality is INSUFFICIENT.
    - Offer plausible options **strictly based on the specific event's details**.
    - Be "single_choice_question" or "multiple_choice_question".

    2) For answer options:
    - Provide exactly 4 options: A, B, C, D. Each option's value **must be a descriptive string** that aligns with the modality being queried (i.e., visual characteristics for visual questions, or audio events for audio questions).
    - Distractor options must be plausible within the event but not aligned with the queried information, or entirely incorrect.
    - For "single_choice_question": exactly one correct descriptive option.
    - For "multiple_choice_question": at least two correct descriptive options.

    3) For explanations:
    - Justify the correct answer(s) by synthesizing information **as if you were observing the video directly** and field the `"gold_reasoning"`.
    - Reasoning must detail inference steps, **explicitly referencing the single event_id and both visual + audio evidence used to align the audio event with its visual manifestation.**
"""

TIMELINE_RECONSTRUCTION_USER_PROMPT = """
video_id: {video_id}
summary: {summary}
events: {events_str}

Task requirements:
    1) Generate N=2 **Timeline Reconstruction** question. The question MUST:
    - Present a list of 4-10 distinct sub-events in a shuffled, non-chronological order. Each sub-event should be explicitly numbered (e.g., "(1) [Description of sub-event A]", "(2) [Description of sub-event B]"). 
    - Each sub-event description should be **concise and focuses on a single, atomic action or observation**.
    - Sub-events should be drawn from **at least 3 different** event_ids.
    - Require the reconstruction of the correct chronological order of these sub-events.
    - Require BOTH visual(e.g., character movements, object appearance/disappearance) and audio(e.g., specific sound effects, spoken time indicators) evidence to determine the correct sequence; single modality is INSUFFICIENT.
    - Be "single_choice_question".

    2) For answer options:
    - Provide exactly 4 options: A, B, C, D. Each option's value **must be a sequence of the sub-event numbers**, joined by " -> ".
    - Provide exactly 1 correct option which represents the correct chronological sequence of the numbered sub-events.
    - Provide exactly 3 distractor options, which must be plausible but incorrect sequences.
    - Ensure all sub-event numbers included in the question are used exactly once in each answer option's sequence.

    3) For explanations:
    - Justify the correct answer(s) by synthesizing information **as if you were observing the video directly** and field the `"gold_reasoning"`.
    - Reasoning must detail inference steps, **explicitly referencing the required event_ids and both visual + audio evidence used to reconstruct the sub-events.**
"""

TOPIC_STANCE_EVOLUTION_SUMMARIZATION_USER_PROMPT = """
video_id: {video_id}
summary: {summary}
events: {events_str}

Task requirements:
    1) Generate N=2 **Topic/Stance Evolution Summarization** question. The question MUST:
    - Focus on summarizing the **evolution or development of a key topic or a character's stance/viewpoint** across multiple relevant events.
    - Involve **at least 3 different event_ids**.
    - Require BOTH visual(e.g., speaker's gestures, on-screen text, changes in setting) and audio(e.g., spoken content, tone shifts, emphasis) evidence to formulate a comprehensive summary; single modality is INSUFFICIENT.
    - Offer plausible options **strictly based on the video's main idea**.
    - Be "single_choice_question" or "multiple_choice_question".

    2) For answer options:
    - Provide exactly 4 options: A, B, C, D. Each option's value **must be a concise, multi-sentence paragraph (2-4 sentences)** describing a potential progression or evolution of the topic/stance across the selected events.
    - Distractor options must be plausible descriptions of an evolution, but either not aligned with the actual progression or entirely incorrect.
    - For "single_choice_question": exactly one correct option.
    - For "multiple_choice_question": at least two correct options.

    3) For explanations:
    - Justify the correct answer(s) by synthesizing information **as if you were observing the video directly** and field the `"gold_reasoning"`.
    - Reasoning must detail inference steps, **explicitly referencing the required event_ids and both visual + audio evidence to support the stated progression or evolution.**
"""

CROSS_EVENT_CAUSALITY_USER_PROMPT = """
video_id: {video_id}
summary: {summary}
events: {events_str}

Task requirements:
    Task requirements:
    1) Generate N=2 **Cross-event Causality Reasoning** question. The question MUST:
    - Choose a specific **"result sub-event"**, which is a localized action or state change within a larger event_id.
    - Ask to identify the preceding event_id(s) and/or specific sub-event(s) within those event_id(s) that most plausibly served as the direct cause or primary contributing factor to the target result sub-event.
    - The causal relationship must span **at least 3 different event_ids**.
    - Require BOTH visual and audio evidence to robustly establish the causal link; single modality is INSUFFICIENT.
    - Offer plausible options **strictly based on the video's main idea**.
    - Be "single_choice_question" or "multiple_choice_question".

    2) For the answer:
    - Provide exactly 4 options: A, B, C, D. Each option should be an event_ids or a descriptive string of specific sub-events within an event_id.
    - Distractor options must be plausible as preceding events/sub-events, but either not align with the actual causal chain, or are entirely incorrect.
    - For "single_choice_question": exactly one correct option.
    - For "multiple_choice_question": at least two correct options.

    3) For explanations:
    - Justify the correct answer(s) by synthesizing information **as if you were observing the video directly** and field the `"gold_reasoning"`.
    - Reasoning must detail inference steps, **explicitly referencing the required event_ids and both visual + audio evidence to support the stated progression or evolution.**
    - Clearly explain **how** the referenced cause event(s)/sub-event(s) led to the state change or outcome observed in the target result sub-event.
"""