"""
Question Type Specialist Prompts for Phase 8 Vision Generation

Each specialist defines how to generate questions for a specific type.
Used in batched prompts to Claude Vision API.
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Sub-Task Type Auto-Detection
# ============================================================================

# Sub-task type keyword mappings
SUB_TASK_TYPE_KEYWORDS = {
    "Human Behavior Understanding": [
        "gesture", "posture", "movement", "reaction", "body language",
        "facial expression", "action", "motion", "position", "stance"
    ],
    "Scene Recognition": [
        "location", "environment", "setting", "venue", "place", "area",
        "background", "surroundings", "scene", "space"
    ],
    "OCR Recognition": [
        "text", "label", "number", "sign", "caption", "word", "letter",
        "display", "screen", "writing", "reading"
    ],
    "Causal Reasoning": [
        "cause", "effect", "result", "because", "leads to", "why", "reason",
        "makes", "triggers", "produces", "creates"
    ],
    "Intent Understanding": [
        "purpose", "goal", "trying to", "about to", "intends", "aims",
        "objective", "planning", "attempting", "wants"
    ],
    "Hallucination": [
        "subtle", "easily misinterpret", "ambiguous", "confuse", "trick",
        "misleading", "deceptive", "unclear"
    ],
    "Multi-Detail Understanding": [
        "sequence", "multiple", "series", "progression", "both", "simultaneously",
        "while", "together", "combination", "and"
    ]
}


def detect_sub_task_types(question: str, answer: str) -> Optional[str]:
    """
    Auto-detect sub-task type based on question and answer content.

    Args:
        question: Question text
        answer: Answer text

    Returns:
        Most relevant sub-task type, or None if no strong match
    """
    combined_text = (question + " " + answer).lower()

    # Score each sub-task type
    scores = {}
    for sub_type, keywords in SUB_TASK_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined_text)
        if score > 0:
            scores[sub_type] = score

    # Return highest scoring type if score >= 2 (require at least 2 keyword matches)
    if scores:
        best_type = max(scores.items(), key=lambda x: x[1])
        if best_type[1] >= 2:
            return best_type[0]

    return None


# Specialist prompt library for all 13 question types
SPECIALIST_PROMPTS = {
    "Needle": """
NEEDLE QUESTIONS (Text Recognition + Audio):
Generate 1-2 questions linking visible text to the audio cue.

CRITICAL: Question MUST require BOTH audio AND visual to answer.

Requirements:
- Identify visible text (signs, labels, numbers, captions, scoreboards)
- Link the text to what's being said/heard in audio cue
- Questions must require EXACT text recall + audio understanding
- Cannot be answered from visual or audio alone

Examples:
✓ "What text is visible on screen when the audio cue '{audio_snippet}' is heard?"
✓ "What number appears on the scoreboard during the phrase '{audio_snippet}'?"
✓ "When the speaker says '{audio_snippet}', what words are displayed on screen?"
✗ "What text is visible?" (no audio requirement - REJECT)
✗ "What does the audio say?" (no visual requirement - REJECT)

⚠️ ANTI-HALLUCINATION CHECK (CRITICAL):
Before generating, verify:
☐ Is the text I'm describing ACTUALLY visible in the frame? (not assumed/invented)
☐ Can I see the exact letters/numbers clearly? (not blurry or partially visible)
☐ Am I inventing text that might be there but isn't clearly shown?

✗ BAD: "Text reading 'CHAMPIONS' is visible" (when text is blurry/unclear)
✓ GOOD: "Scoreboard displays '127' in large white numbers"

ONLY describe text you can ACTUALLY READ in the frame.

FORBIDDEN HEDGING LANGUAGE:
✗ NO "appears to", "seems to", "looks like", "could be", "may be", "might be"
✗ NO "suggests", "likely", "probably", "possibly"
✓ USE DEFINITIVE: "is", "reads", "displays", "shows", "contains"

FORBIDDEN:
- No pronouns (he/she/they/him/her) - use descriptors (the person, the individual)
- No "at what time" questions
- No names in questions or answers
""",

    "Audio-Visual Stitching": """
AUDIO-VISUAL STITCHING:
Generate 1-2 questions linking this visual moment to the audio cue.

Audio cue at this timestamp: "{audio_cue}"

CRITICAL: Question MUST require BOTH audio AND visual to answer.

═══════════════════════════════════════════════════════════════════════════════
AUDIO MODALITY DIVERSITY REQUIREMENTS (MANDATORY)
═══════════════════════════════════════════════════════════════════════════════

ACROSS YOUR QUESTIONS, YOU MUST USE ALL AUDIO MODALITIES:
If generating 2 questions, at minimum:
- 1 question using SPEECH/TRANSCRIPT as primary cue
- 1 question using NON-SPEECH (music/sound/silence) as primary cue

If generating 3+ questions, at minimum:
- 1 SPEECH-based question
- 1 MUSIC/TEMPO-based question
- 1 SOUND EFFECT-based question

AUDIO MODALITY EXAMPLES:

1️⃣ SPEECH-BASED (mention specific words/phrases from transcript):
   ✓ "When you hear '{audio_snippet}', what action is the person performing?"
   ✓ "What object is visible when the phrase '{audio_snippet}' is spoken?"

2️⃣ MUSIC-BASED (tempo/tone changes, music starts/stops):
   ✓ "What action happens when the music tempo increases from 80 to 120 BPM?"
   ✓ "What visual transition occurs as background music shifts to dramatic?"

3️⃣ SOUND EFFECT-BASED (impact, whoosh, click, mechanical sounds):
   ✓ "When you hear the impact sound, what collision is visible?"
   ✓ "What motion produces the whoosh sound at this moment?"

4️⃣ CROWD SOUND-BASED (applause, cheering, crowd reaction):
   ✓ "What visual change occurs when the applause starts?"
   ✓ "What is happening when you hear the crowd cheering?"

5️⃣ SILENCE-BASED (dramatic pauses, audio gaps, scene cuts):
   ✓ "What visual change occurs during the 2-second pause in audio?"
   ✓ "What moment corresponds with the silence?"

DIVERSITY CHECK - Before finalizing questions:
☐ Have I used at least 1 speech-based cue?
☐ Have I used at least 1 non-speech cue (music/sound/silence)?
☐ Am I using DIFFERENT audio modalities across my questions (not all speech)?

═══════════════════════════════════════════════════════════════════════════════

Requirements:
- Link what's happening visually to what's being said/heard
- Use ALL audio elements: speech, music, sound effects, silence, crowd sounds
- Answer must reference both modalities explicitly
- Cannot be answered from audio or visual alone

AUDIO_CUE FORMAT EXAMPLES:
The audio_cue field will contain:
- Speech only: "'Hello world' at 0:15"
- Speech + music: "'Hello world'; intro music (tempo: 140 BPM)"
- Speech + sound: "'Impact!'; high intensity impact sound effect"
- Music only: "intro music (tempo: 140 BPM)"
- Music change: "Music tempo increase by 30 BPM"
- Multiple: "'Watch this'; high intensity impact sound effect; medium applause"
- Silence: "Dramatic pause (2.5s silence)"

Examples (DIVERSE AUDIO TYPES):

SPEECH-BASED:
✓ "When you hear '{audio_snippet}', what action is the person performing?"
✓ "What visual change occurs when the speaker says '{audio_snippet}'?"

MUSIC-BASED:
✓ "What action is the person in blue performing when the music tempo increases?"
✓ "What visual transition occurs when the background music changes from slow to fast?"

SOUND EFFECT-BASED:
✓ "When you hear the impact sound, what is happening visually?"
✓ "What action produces the whoosh sound at this moment?"

CROWD SOUND-BASED:
✓ "What visual change occurs immediately after the applause starts?"
✓ "What is the person doing when you hear the crowd cheering?"

SILENCE-BASED:
✓ "What visual moment corresponds with the dramatic silence?"
✓ "What is visible during the pause in audio?"

⚠️ ANTI-HALLUCINATION CHECK (CRITICAL):
Before generating, verify:
☐ Is the audio-visual connection I'm describing ACTUALLY present? (not assumed)
☐ Am I inventing synchronization that doesn't exist?
☐ Can I point to specific audio AND visual evidence that align?

✗ BAD: "Person reacts to explosion sound" (when audio has no explosion)
✓ GOOD: "Person moves right as narrator says 'move to position'"

ONLY describe audio-visual connections you can ACTUALLY OBSERVE.

FORBIDDEN HEDGING LANGUAGE:
✗ NO "appears to", "seems to", "looks like", "could be", "may be", "might be"
✗ NO "suggests", "likely", "probably", "possibly"
✓ USE DEFINITIVE: "is", "shows", "displays", "contains", "demonstrates"

FORBIDDEN:
✗ "What is happening in the video?" (no audio requirement - REJECT)
✗ "What is being said?" (no visual requirement - REJECT)
✗ "What does the speaker say?" (audio only - REJECT)
- No pronouns (he/she/they/him/her) - use "the person", "the individual"
- No names - use descriptors ("person in blue shirt", "individual on left")
- No "at what time" questions
""",

    "Temporal Understanding": """
TEMPORAL UNDERSTANDING (Audio + Visual Timing):
Generate 1-2 questions about timing, sequence, or progression.

CRITICAL: Question MUST require BOTH audio AND visual to answer.

Requirements:
- Link temporal events to audio cue: "before/after X is said, what do you see?"
- Use ALL audio types as temporal anchors: speech, music, sounds, silence
- Reference this moment in video timeline using audio + visual anchors
- Answers use temporal language (before, after, during, when, first, next)
- Cannot answer with visual timing alone - must use audio as anchor

AUDIO_CUE FORMAT EXAMPLES:
The audio_cue will contain various audio elements:
- Speech: "'Let's begin' at 0:20"
- Music tempo change: "Music tempo increase by 30 BPM"
- Sound effect: "high intensity impact sound effect"
- Silence: "Dramatic pause (2.5s silence)"
- Combined: "'Watch this'; background music (tempo: 150 BPM); high intensity whoosh sound effect"

Examples (DIVERSE AUDIO ANCHORS):

SPEECH ANCHORS:
✓ "What visual action occurs immediately after the audio cue '{audio_snippet}' ends?"
✓ "When you hear '{audio_snippet}', what change is happening on screen?"

MUSIC ANCHORS:
✓ "What do you see happening when the background music shifts from slow to fast?"
✓ "What visual change occurs during the music tempo increase?"

SOUND EFFECT ANCHORS:
✓ "What action happens immediately before the impact sound?"
✓ "When you hear the whoosh sound, what movement is occurring?"

SILENCE ANCHORS:
✓ "What visual change happens during the pause in audio?"
✓ "What is happening on screen when the silence begins?"

⚠️ ANTI-HALLUCINATION CHECK (CRITICAL):
Before generating, verify:
☐ Is the action/change I'm describing ACTUALLY visible in the frame? (not assumed)
☐ Am I inventing motion or progression that isn't clearly shown?
☐ Can I point to specific visual evidence of this timing/sequence?

✗ BAD: "Person moves from left to right" (when person is stationary)
✓ GOOD: "Person's position shifts from left side to center of frame"

ONLY describe temporal changes you can ACTUALLY SEE happening.

FORBIDDEN HEDGING LANGUAGE:
✗ NO "appears to", "seems to", "looks like", "could be", "may be", "might be"
✗ NO "suggests", "likely", "probably", "possibly"
✓ USE DEFINITIVE: "is", "changes", "transitions", "progresses", "shows"

FORBIDDEN:
✗ "What happens at 2:30?" (timestamp question - FORBIDDEN)
✗ "What happens first?" (no audio anchor - REJECT)
✗ "What color changes?" (no temporal + audio link - REJECT)
- No "at what time" or "what timestamp" questions
- No pronouns (he/she/they) - use descriptors
- No names
""",

    "Sequential": """
SEQUENTIAL (Multi-Step Progression with Audio Anchors):
Generate 1-2 questions about ordered steps or multi-part actions.

CRITICAL: Question MUST require BOTH audio AND visual to answer.

═══════════════════════════════════════════════════════════════
⚠️ HARD REQUIREMENT: 3+ STEPS MINIMUM (AUTO-REJECT IF < 3)
═══════════════════════════════════════════════════════════════

Your ANSWER MUST describe AT LEAST 3 distinct steps/actions.

STEP COUNTING TEST (apply before finalizing):
1. Count distinct actions/states in your answer
2. If count < 3 → REJECT and rewrite with more steps
3. Only if count ≥ 3 → ACCEPT

VALID (3+ steps):
✓ Q: "What sequence of actions occurs from when '{audio_snippet}' is heard?"
  A: "First, the person picks up the tool. Then, the person positions it over the material. Next, the person applies pressure. Finally, the person removes the tool." (4 steps)

✓ Q: "List the steps shown between '{audio_start}' and '{audio_end}'."
  A: "Step 1: Open container. Step 2: Pour liquid. Step 3: Seal container." (3 steps)

✓ Q: "Track what happens when '{audio_snippet}' occurs."
  A: "The person begins by reaching for the object, continues by grasping it, then lifts it upward." (3 steps)

INVALID (< 3 steps):
✗ Q: "What happens when '{audio_snippet}' is heard?"
  A: "Person picks up object, then sets it down." (2 steps) → REJECT

✗ Q: "Describe the action during '{audio_snippet}'."
  A: "Person performs the task." (1 step) → REJECT

REQUIRED SEQUENTIAL KEYWORDS (use at least 2 of these in your answer):
• "first... then... next... finally"
• "step 1... step 2... step 3"
• "initially... subsequently... ultimately"
• "begins by... continues with... concludes with"
• "starts with... proceeds to... ends with"

Requirements:
- Questions about step-by-step processes anchored to audio cues
- Use audio to mark sequence points: "when X is said, which steps are shown?"
- Focus on order and progression linked to audio + visual
- MUST require analyzing transitions between steps, not just listing them

MULTI-STEP REASONING TEMPLATE:
Questions should require the answerer to:
1. Identify/observe the initial state or action
2. Track how it changes or progresses through intermediate steps
3. Analyze the final state or result
4. Explain the causal or temporal relationships between steps

⚠️ ANTI-HALLUCINATION CHECK (CRITICAL):
Before generating, verify:
☐ Are all the steps I'm describing ACTUALLY visible in sequence? (not assumed)
☐ Am I inventing intermediate steps that aren't clearly shown?
☐ Can I point to specific moments where each step occurs?

✗ BAD: "Person picks up object, rotates it, then sets it down" (when only pickup is visible)
✓ GOOD: "Person extends hand toward object, grasps it, raises it to chest level"

ONLY describe steps you can ACTUALLY SEE in the frame sequence.

FORBIDDEN HEDGING LANGUAGE:
✗ NO "appears to", "seems to", "looks like", "could be", "may be", "might be"
✗ NO "suggests", "likely", "probably", "possibly"
✓ USE DEFINITIVE: "is", "performs", "moves", "transitions", "progresses"

FORBIDDEN:
- No pronouns (he/she/they) - use "the person", "the individual"
- No names
- No "at what time" questions
- No simple listing without explaining transitions/causality
- Answers with fewer than 3 distinct steps (AUTO-REJECT)
""",

    "Subscene": """
SUBSCENE (Continuous Action + Audio):
Generate 1-2 questions about the ongoing continuous action within this scene.

CRITICAL: Question MUST require BOTH audio AND visual to answer.

Requirements:
- Link the action phase/stage to the audio cue
- Questions about stage of action when audio occurs
- Focus on micro-level progression tied to what's heard
- Intermediate states anchored by audio

Examples:
✓ "What stage of the jump shot is shown when you hear '{audio_snippet}'?"
✓ "When the sound of '{audio_snippet}' occurs, what motion phase is visible?"
✓ "At what point in the throwing action does the audio cue '{audio_snippet}' happen?"
✗ "What stage is shown?" (no audio requirement - REJECT)
✗ "Is this the beginning or end?" (no audio anchor - REJECT)
✗ "What sport is being played?" (too general + no audio - REJECT)

⚠️ ANTI-HALLUCINATION CHECK (CRITICAL):
Before generating, verify:
☐ Is the action phase/stage I'm describing ACTUALLY visible? (not assumed)
☐ Am I inventing motion phases that aren't clearly shown?
☐ Can I see the specific stage of this continuous action?

✗ BAD: "Mid-flight phase of jump" (when person is on ground)
✓ GOOD: "Preparation phase with knees bent before jump"

ONLY describe action stages you can ACTUALLY SEE.

FORBIDDEN HEDGING LANGUAGE:
✗ NO "appears to", "seems to", "looks like", "could be", "may be", "might be"
✗ NO "suggests", "likely", "probably", "possibly"
✓ USE DEFINITIVE: "is", "shows", "displays", "demonstrates"

FORBIDDEN:
- No pronouns (he/she/they) - use "the person in [descriptor]"
- No names
- No "at what time" questions
""",

    "General Holistic Reasoning": """
GENERAL HOLISTIC REASONING (Scene + Audio Context):
Generate 1-2 questions requiring understanding of the overall scene + audio.

CRITICAL: Question MUST require BOTH audio AND visual to answer.

Requirements:
- Questions about complete scene/situation using audio + visual context
- Integrate audio context (music, sounds, speech) with visual elements
- Require broad understanding of both modalities
- "Big picture" questions that need both audio and visual to answer

Examples:
✓ "When you hear '{audio_snippet}', what type of event is taking place?"
✓ "What overall atmosphere is created by the combination of what you see and hear?"
✓ "Based on the visual scene and the audio cue '{audio_snippet}', what activity is occurring?"
✗ "What is happening?" (no audio requirement - REJECT)
✗ "What type of event is this?" (visual only - REJECT)
✗ "What color dominates the scene?" (visual detail only - REJECT)

⚠️ ANTI-HALLUCINATION CHECK (CRITICAL):
Before generating, verify:
☐ Is the overall scene/situation I'm describing ACTUALLY evident? (not assumed)
☐ Am I inventing context or atmosphere not supported by both modalities?
☐ Can I point to specific audio AND visual elements that establish this context?

✗ BAD: "Tense competitive atmosphere" (when no evidence of tension/competition)
✓ GOOD: "Indoor practice session based on gym setting and casual instruction"

ONLY describe contexts you can ACTUALLY INFER from observable audio-visual evidence.

FORBIDDEN HEDGING LANGUAGE:
✗ NO "appears to", "seems to", "looks like", "could be", "may be", "might be"
✗ NO "suggests", "likely", "probably", "possibly"
✓ USE DEFINITIVE: "is", "shows", "indicates", "demonstrates"

FORBIDDEN:
- No pronouns (he/she/they) - use "the person", "the individual"
- No names
- No "at what time" questions
""",

    "Inference": """
INFERENCE (Audio-Visual Causal Reasoning):
Generate 1-2 questions requiring reasoning beyond literal observation.

CRITICAL: Question MUST require BOTH audio AND visual to answer.

═══════════════════════════════════════════════════════════════
⚠️ HARD REQUIREMENT: MUST ASK WHY/HOW (AUTO-REJECT IF MISSING)
═══════════════════════════════════════════════════════════════

Your QUESTION MUST contain one of these CAUSAL keywords:
• "WHY" (motivation, reason, purpose)
• "HOW" (mechanism, process, method)
• "WHAT causes" / "WHAT leads to"
• "WHAT evidence suggests" / "WHAT indicates"

CAUSAL KEYWORD TEST (apply before finalizing):
1. Does question contain WHY/HOW/WHAT causes/WHAT evidence?
2. If NO → REJECT and rewrite with causal framing
3. Only if YES → ACCEPT

VALID (has causal keyword):
✓ Q: "WHY does the person perform this action when '{audio_snippet}' is heard?"
  A: "The person performs this action because the audio instructs them to do so, and the visual shows the necessary tool is available."

✓ Q: "HOW does the audio cue '{audio_snippet}' relate to the visual change?"
  A: "The audio describes the process step-by-step, which directly corresponds to the visual progression shown."

✓ Q: "WHAT evidence suggests the person's intent during '{audio_snippet}'?"
  A: "The person glances at the target object while the audio mentions it, suggesting intentional focus."

✓ Q: "WHAT causes the visible change when '{audio_snippet}' occurs?"
  A: "The audio instruction triggers the person to initiate the action, causing the visual transformation."

INVALID (no causal keyword):
✗ Q: "What action is performed when '{audio_snippet}' is heard?"
  A: "Person picks up object." → REJECT (descriptive only, no inference)

✗ Q: "What is visible during '{audio_snippet}'?"
  A: "Red box on table." → REJECT (observation only, no reasoning)

✗ Q: "What do you see and hear?"
  A: "Person moving, music playing." → REJECT (no causal connection required)

✗ Q: "Describe what happens."
  A: "Person performs task." → REJECT (no WHY/HOW)

FORBIDDEN QUESTION PATTERNS (auto-reject):
• "What do you see..." (no inference)
• "What is happening..." (descriptive)
• "What action..." (no reasoning)
• "Describe the..." (no causality)
• "What appears..." (observation only)
• "What is shown..." (no causal analysis)

Requirements:
- Infer implied information from audio + visual cues together
- Deduce emotions, intentions, relationships using both modalities
- Explain WHY something happens or HOW something works
- "Read between the lines" using what you see AND hear
- Cannot infer from visual or audio alone
- Must require explaining causality, not just describing

CAUSAL REASONING TEMPLATE:
Questions should ask WHY or HOW:
1. Why does an action occur? (motivation/intention)
2. How does X cause Y? (mechanism)
3. What evidence suggests Z? (justification)
4. Why choose A over B? (reasoning)

⚠️ ANTI-HALLUCINATION CHECK (CRITICAL):
Before generating, verify:
☐ Is the causal relationship I'm inferring ACTUALLY supported by evidence? (not fabricated)
☐ Am I inventing motivations/intentions not evident in audio-visual cues?
☐ Can I point to specific observable evidence supporting this inference?

✗ BAD: "Person avoids object because it's dangerous" (when no danger evident)
✓ GOOD: "Person reaches for object after narrator mentions its importance"

ONLY make inferences SUPPORTED by observable audio-visual evidence.

FORBIDDEN HEDGING LANGUAGE:
✗ NO "appears to", "seems to", "looks like", "could be", "may be", "might be"
✗ NO "suggests", "likely", "probably", "possibly" (except in answer explanations)
✓ USE DEFINITIVE: "is", "shows", "indicates", "demonstrates", "evidences"

Note: When ANSWERING WHY/HOW questions, terms like "suggests" and "indicates" are acceptable
for explaining reasoning (e.g., "The visual evidence suggests..."). But avoid in descriptions.

FORBIDDEN:
- No pronouns (he/she/they) - use "the person", "the individual"
- No names - use descriptors
- No "at what time" questions
- No questions without WHY/HOW/WHAT causes/WHAT evidence (AUTO-REJECT)
- No purely descriptive questions (must require causal reasoning)
""",

    "Context": """
CONTEXT (Audio-Visual Setting):
Generate 1-2 questions about the setting or situational context.

CRITICAL: Question MUST require BOTH audio AND visual to answer.

Requirements:
- Identify context using ALL audio elements (ambient sounds, music, speech, crowd, silence) + visual clues
- Location, environment, situation type inferred from both modalities
- Scene setting determined by what you see AND hear
- Cannot determine context from visual or audio alone

AUDIO_CUE FORMAT EXAMPLES:
The audio_cue may contain various context indicators:
- Speech: "'Welcome to the stadium' at 1:45"
- Ambient sound: "medium traffic sounds; distant car horn"
- Music: "upbeat music (tempo: 160 BPM)"
- Crowd: "loud applause; medium crowd cheering"
- Multiple: "'Let's begin'; background music (tempo: 130 BPM); low crowd murmuring"

Examples (DIVERSE AUDIO CONTEXT):

SPEECH CONTEXT:
✓ "Based on what you see and the audio cue '{audio_snippet}', what type of venue is this?"
✓ "When you hear '{audio_snippet}' and see the surroundings, is this a professional or casual setting?"

MUSIC CONTEXT:
✓ "What type of setting is indicated by the upbeat music and the visible environment?"
✓ "Based on the background music tempo and the visual scene, what activity is taking place?"

CROWD SOUND CONTEXT:
✓ "What type of venue is indicated by the applause and the visible seating?"
✓ "Based on the crowd cheering and the visual layout, what event is this?"

AMBIENT SOUND CONTEXT:
✓ "What environment is indicated by the traffic sounds and the visible surroundings?"
✓ "Based on the nature sounds and the visual setting, where is this taking place?"

SILENCE CONTEXT:
✓ "What type of moment is indicated by the silence and the visual expressions?"
✓ "Based on the lack of audio and the visual scene, what atmosphere is created?"

⚠️ ANTI-HALLUCINATION CHECK (CRITICAL):
Before generating, verify:
☐ Is the setting/context I'm describing ACTUALLY evident from audio-visual cues? (not assumed)
☐ Am I inventing venue details not supported by visible/audible evidence?
☐ Can I point to specific audio AND visual elements confirming this context?

✗ BAD: "Professional stadium" (when visual shows basic gym)
✓ GOOD: "Indoor gym based on wooden floor and background echo in audio"

ONLY describe contexts SUPPORTED by observable audio-visual evidence.

FORBIDDEN HEDGING LANGUAGE:
✗ NO "appears to", "seems to", "looks like", "could be", "may be", "might be"
✗ NO "suggests", "likely", "probably", "possibly"
✓ USE DEFINITIVE: "is", "shows", "indicates", "demonstrates"

FORBIDDEN:
✗ "What type of venue is this?" (visual only - REJECT)
✗ "Is this indoor or outdoor?" (visual only - REJECT)
✗ "What is the lighting like?" (visual detail only - REJECT)
- No pronouns (he/she/they) - use descriptors
- No names
- No "at what time" questions
""",

    "Referential Grounding": """
REFERENTIAL GROUNDING (Spatial + Audio):
Generate 1-2 questions about object positions, locations, and attributes.

CRITICAL: Question MUST require BOTH audio AND visual to answer.

Requirements:
- Link spatial relationships to audio cue
- Reference objects/people mentioned or occurring during audio
- Use position words (left, right, behind, front, near, holding, wearing)
- Must use audio to identify WHICH object/person being asked about

Examples:
✓ "What color jersey is worn by the person who you hear speaking in '{audio_snippet}'?"
✓ "When '{audio_snippet}' is heard, what object is the person on the left holding?"
✓ "What is positioned behind the individual mentioned in the audio cue '{audio_snippet}'?"
✗ "What color is the jersey?" (no audio link - REJECT)
✗ "What is the person holding?" (no audio anchor - REJECT)
✗ "How many people are visible?" (counting, no audio - REJECT)

⚠️ ANTI-HALLUCINATION CHECK (CRITICAL):
Before generating, verify:
☐ Are the objects/positions I'm describing ACTUALLY visible? (not assumed)
☐ Am I inventing spatial relationships that aren't clearly shown?
☐ Can I see the exact colors/attributes I'm mentioning?

✗ BAD: "Person holds blue object" (when object color is unclear)
✓ GOOD: "Person on left side wears red jersey with visible number"

ONLY describe positions and attributes you can ACTUALLY SEE.

FORBIDDEN HEDGING LANGUAGE:
✗ NO "appears to", "seems to", "looks like", "could be", "may be", "might be"
✗ NO "suggests", "likely", "probably", "possibly"
✓ USE DEFINITIVE: "is", "wears", "holds", "positioned at", "located"

FORBIDDEN:
- No pronouns (he/she/they/him/her) - use "the person", "the individual"
- No names - use descriptors ("person in blue", "individual on left")
- No "at what time" questions
""",

    "Counting": """
COUNTING (with Audio Context):
Generate 1-2 questions requiring counting visible elements.

CRITICAL: Question MUST require BOTH audio AND visual to answer.

Requirements:
- Count objects/people mentioned or occurring during audio cue
- Use audio to specify WHAT to count or WHEN to count
- Answer must be specific number grounded in both modalities
- Cannot answer with visual count alone - need audio context

Examples:
✓ "How many people are visible when the audio cue '{audio_snippet}' is heard?"
✓ "When '{audio_snippet}' occurs, how many objects are on the surface shown?"
✓ "How many distinct sounds do you hear while seeing the count of items shown?"
✗ "How many people are visible?" (no audio requirement - REJECT)
✗ "How many windows are there?" (visual only - REJECT)
✗ "Are there multiple people?" (yes/no + no audio - REJECT)

⚠️ ANTI-HALLUCINATION CHECK (CRITICAL):
Before generating, verify:
☐ Can I count the exact number clearly? (not estimating from partial view)
☐ Am I inventing additional items that aren't fully visible?
☐ Are all the items I'm counting actually distinct and countable?

✗ BAD: "5 people visible" (when only 3 are clearly shown, 2 partially obscured)
✓ GOOD: "3 people clearly visible in frame when audio occurs"

ONLY count items you can ACTUALLY SEE completely.

FORBIDDEN HEDGING LANGUAGE:
✗ NO "appears to", "seems to", "looks like", "could be", "may be", "might be"
✗ NO "suggests", "likely", "probably", "possibly", "approximately"
✓ USE DEFINITIVE: "is", "are", "shows", "displays", "contains" + exact numbers

FORBIDDEN:
- No pronouns (he/she/they) - use descriptors
- No names
- No "at what time" questions
""",

    "Comparative": """
COMPARATIVE (Audio-Visual Contrasts Over Time):
Generate 1-2 questions comparing elements across time or showing contrasts.

CRITICAL: Question MUST require BOTH audio AND visual to answer.

═══════════════════════════════════════════════════════════════
⚠️ HARD REQUIREMENT: 3+ COMPARISON POINTS (AUTO-REJECT IF < 3)
═══════════════════════════════════════════════════════════════

Your ANSWER MUST compare AT LEAST 3 distinct states/moments/points.

COMPARISON POINT TEST (apply before finalizing):
1. Count distinct states/moments compared in answer
2. If count < 3 → REJECT and rewrite comparing more points
3. Only if count ≥ 3 → ACCEPT

VALID (3+ comparison points):
✓ Q: "How does the object's position change from start to middle to end during '{audio_snippet}'?"
  A: "At the start (when you hear X), object is on the left. At the middle (when you hear Y), object has moved to center. At the end (when you hear Z), object is on the right." (3 points)

✓ Q: "Track the color progression across the sequence anchored by '{audio_snippet}'."
  A: "At moment A: red. At moment B: orange. At moment C: yellow. At moment D: green." (4 points)

✓ Q: "Compare the person's height position at three stages during '{audio_snippet}'."
  A: "Initially: crouching low. Midway: rising to half-height. Finally: standing fully upright." (3 points)

INVALID (< 3 comparison points):
✗ Q: "What's different between start and end during '{audio_snippet}'?"
  A: "At start: X is red. At end: X is blue." (2 points) → REJECT

✗ Q: "Compare before and after '{audio_snippet}'."
  A: "Before: closed. After: open." (2 points) → REJECT

✗ Q: "How does it change?"
  A: "It transitions from one state to another." (2 points) → REJECT

FORBIDDEN PATTERNS (auto-reject):
• "before and after" (only 2 points)
• "start vs end" (only 2 points)
• "initially vs finally" (only 2 points)
• "compare X and Y" (only 2 items, need 3+)

REQUIRED COMPARISON KEYWORDS (use at least 2 in your answer):
• "at moment A... at moment B... at moment C"
• "from start... to middle... to end"
• "initially... midway through... ultimately"
• "across 3 stages" / "through 4 phases"
• "first point... second point... third point"
• "at the beginning... at the midpoint... at the conclusion"

Requirements:
- Compare visual elements at 3+ points in time (not just start vs end)
- Use audio cues to anchor comparison points
- Compare audio characteristics + visual characteristics together
- Must include quantitative or qualitative degree of change
- Must analyze rate of change (gradual vs sudden)

MULTI-STEP COMPARISON TEMPLATE:
Questions should require the answerer to:
1. Identify the initial state (when audio starts)
2. Observe intermediate states (during audio)
3. Compare final state (when audio ends)
4. Quantify or qualify the degree of change
5. Analyze the rate/pattern of change

⚠️ ANTI-HALLUCINATION CHECK (CRITICAL):
Before generating, verify:
☐ Are all 3+ comparison points I'm describing ACTUALLY visible across the sequence? (not assumed)
☐ Am I inventing intermediate states not clearly shown?
☐ Can I see the specific differences at each comparison point?

✗ BAD: "Object moves left → center → right" (when only left and right positions visible)
✓ GOOD: "Color changes from red (start) → orange (middle frame) → yellow (end frame)"

ONLY compare states you can ACTUALLY OBSERVE at 3+ distinct moments.

FORBIDDEN HEDGING LANGUAGE:
✗ NO "appears to", "seems to", "looks like", "could be", "may be", "might be"
✗ NO "suggests", "likely", "probably", "possibly"
✓ USE DEFINITIVE: "is", "changes to", "transitions from", "progresses to"

FORBIDDEN:
- No pronouns (he/she/they) - use "the person who...", "the individual in..."
- No names
- No "at what time" questions
- No simple binary comparisons (before/after only) - MUST have 3+ points
- No comparisons without quantification or qualification
- Answers with fewer than 3 distinct comparison points (AUTO-REJECT)
""",

    "Object Interaction Reasoning": """
OBJECT INTERACTION REASONING (Audio-Visual Interactions):
Generate 1-2 questions about how people interact with objects.

CRITICAL: Question MUST require BOTH audio AND visual to answer.

Requirements:
- Link person-object interaction to audio cue
- Questions about how objects are used while audio occurs
- Audio context helps identify interaction type or timing
- Cannot determine interaction from visual or audio alone

Examples:
✓ "When you hear '{audio_snippet}', how is the person using the object visible?"
✓ "What is being done with the object held when the sound of '{audio_snippet}' occurs?"
✓ "Based on what you see and hear, in what manner is the interaction happening?"
✗ "How is the object being used?" (no audio requirement - REJECT)
✗ "What is the person doing?" (no audio anchor - REJECT)
✗ "Is an object being held?" (yes/no + no audio - REJECT)

⚠️ ANTI-HALLUCINATION CHECK (CRITICAL):
Before generating, verify:
☐ Is the person-object interaction I'm describing ACTUALLY happening? (not assumed)
☐ Am I inventing physical contact or manipulation not clearly visible?
☐ Can I see the actual interaction between person and object?

✗ BAD: "Person grips object tightly" (when object is just near person's hand)
✓ GOOD: "Person's hand is positioned next to object"

ONLY describe interactions you can ACTUALLY SEE with clear physical contact/manipulation.

FORBIDDEN HEDGING LANGUAGE:
✗ NO "appears to", "seems to", "looks like", "could be", "may be", "might be"
✗ NO "suggests", "likely", "probably", "possibly"
✓ USE DEFINITIVE: "holds", "grabs", "pushes", "moves", "positions"

FORBIDDEN:
- No pronouns (he/she/they) - use "the person", "the individual"
- No names
- No "at what time" questions
""",

    "Tackling Spurious Correlations": """
TACKLING SPURIOUS CORRELATIONS (Audio-Visual Anomalies):
Generate 1-2 questions about unusual, unexpected, or anomalous elements.

CRITICAL: Question MUST require BOTH audio AND visual to answer.

Requirements:
- Identify mismatches between audio and visual expectations
- Challenge assumptions using both modalities
- Look for audio-visual contradictions or surprises
- Designed to catch models assuming typical patterns

Examples:
✓ "When you hear '{audio_snippet}', what unexpected visual element contradicts it?"
✓ "What is unusual about what you see compared to what you hear in '{audio_snippet}'?"
✓ "Based on the audio cue '{audio_snippet}', what visual element is surprising or out-of-place?"
✗ "What is unusual visually?" (no audio contrast - REJECT)
✗ "What object is unexpected?" (no audio comparison - REJECT)
✗ "What color contradicts expectations?" (visual only - REJECT)

⚠️ ANTI-HALLUCINATION CHECK (CRITICAL):
Before generating, verify:
☐ Is the audio-visual mismatch I'm describing ACTUALLY present? (not fabricated)
☐ Am I inventing contradictions that don't exist?
☐ Can I point to specific audio AND visual elements that genuinely conflict?

✗ BAD: "Audio mentions fire but no fire visible" (when fire IS actually visible)
✓ GOOD: "Audio says 'blue object' but visual shows red object"

ONLY describe contradictions/anomalies that ACTUALLY EXIST in the audio-visual pairing.

FORBIDDEN HEDGING LANGUAGE:
✗ NO "appears to", "seems to", "looks like", "could be", "may be", "might be"
✗ NO "suggests", "likely", "probably", "possibly"
✓ USE DEFINITIVE: "contradicts", "mismatches", "conflicts with", "differs from"

FORBIDDEN:
- No pronouns (he/she/they) - use descriptors
- No names
- No "at what time" questions
""",
}


# Multi-frame cluster prompts for temporal questions
MULTI_FRAME_PROMPTS = {
    "Temporal Progression": """
TEMPORAL PROGRESSION (Multi-Frame Sequence):
You are viewing {num_frames} consecutive frames spanning {duration:.1f} seconds.

Frame sequence: {frame_list}

Audio progression:
- Start ({start_time:.1f}s): "{audio_start}"
- End ({end_time:.1f}s): "{audio_end}"

Generate 1-2 questions about visual changes/progression across this sequence.

CRITICAL: Questions must require viewing ALL frames to answer.

Requirements:
- Use audio cues as temporal anchors ("between when X is said and Y is said")
- Describe visual change/progression from start to end
- Answer cannot be determined from any single frame
- Answer must describe the progression/transformation

Examples:
✓ "Between when you hear '{audio_start_snippet}' and '{audio_end_snippet}', how does the scene composition change?"
✓ "What visual progression occurs from the moment '{audio_start_snippet}' is heard to when '{audio_end_snippet}' plays?"
✓ "How does the spatial arrangement change between these audio cues: '{audio_start_snippet}' → '{audio_end_snippet}'?"

FORBIDDEN:
- No timestamp references ("at 2:30", "at the start")
- No pronouns (he/she/they) - use "the person", "the individual"
- No names
- No questions answerable from a single frame
""",

    "Sequential Action": """
SEQUENTIAL ACTION (Multi-Frame Steps):
You are viewing {num_frames} consecutive frames showing a sequence of actions.

Audio anchors this sequence timing:
- "{audio_start}" (start)
- "{audio_end}" (end)

Generate 1 question about the ordered sequence of actions.

Requirements:
- Identify distinct action steps across frames
- Use audio to define sequence boundaries
- Answer lists steps in chronological order
- Requires all frames to determine full sequence

Examples:
✓ "What sequence of actions occurs between when you hear '{audio_start_snippet}' and '{audio_end_snippet}'?"
✓ "List the steps performed from the audio cue '{audio_start_snippet}' to '{audio_end_snippet}' in order."

FORBIDDEN:
- No timestamp questions
- No pronouns, no names
- No questions about single actions (must be multi-step)
""",

    "State Transformation": """
STATE TRANSFORMATION (Multi-Frame Change):
You are viewing {num_frames} frames showing an object/scene transformation.

Initial state ({start_time:.1f}s): "{audio_start}"
Final state ({end_time:.1f}s): "{audio_end}"

Generate 1 question about how the object/scene transforms.

Requirements:
- Identify clear state change (position, orientation, configuration, presence)
- Use audio anchors for before/after states
- Answer describes the transformation
- Transformation not visible in any single frame

Examples:
✓ "What transformation occurs between when '{audio_start_snippet}' is heard and '{audio_end_snippet}' plays?"
✓ "How does the object's configuration change from '{audio_start_snippet}' to '{audio_end_snippet}'?"
✓ "What state change happens between these audio moments: '{audio_start_snippet}' → '{audio_end_snippet}'?"

FORBIDDEN:
- No timestamp references
- No pronouns, no names
- No questions about static properties (must show change)
"""
}


def build_specialist_prompt(frame_data: Dict, config: Dict) -> str:
    """
    Build batched specialist prompt for a single frame.

    Combines multiple specialist prompts into one comprehensive prompt
    for Claude Vision API.

    Args:
        frame_data: Frame information including:
            - timestamp: float
            - question_types: List[str]
            - audio_cue: str
            - priority: float
        config: PHASE8_CONFIG dict

    Returns:
        Complete prompt string for Claude Vision API
    """
    question_types = frame_data.get('question_types', [])
    timestamp = frame_data.get('timestamp', 0)
    audio_cue = frame_data.get('audio_cue', '')
    priority = frame_data.get('priority', 0.5)

    # Determine target questions based on priority
    if priority >= 0.90:
        target_questions = config['questions_per_frame']['high_priority']
    elif priority >= 0.75:
        target_questions = config['questions_per_frame']['medium_priority']
    else:
        target_questions = config['questions_per_frame']['low_priority']

    # Build prompt header
    prompt = f"""You are analyzing a video frame at {timestamp:.1f} seconds.

AUDIO CUE at this timestamp (includes speech, music, sounds, and silence):
{audio_cue}

═══════════════════════════════════════════════════════════════════════════════
⚠️  CRITICAL RULE #1: ZERO HEDGING LANGUAGE ⚠️
═══════════════════════════════════════════════════════════════════════════════

Your answers will be REJECTED if they contain ANY of these words:
❌ "appears" / "appears to" / "appear" / "appear to"
❌ "seems" / "seems to" / "seem" / "seem to"
❌ "looks like" / "look like"
❌ "might" / "may" / "could"
❌ "possibly" / "potentially"
❌ "suggests" / "suggesting" / "indicate" / "indicating"

✅ CORRECT: "The object is a LEGO model"
❌ WRONG: "The object appears to be a LEGO model"

✅ CORRECT: "The figure represents a warrior"
❌ WRONG: "The figure appears to represent a warrior"

✅ CORRECT: "The visual shows the figure positioned near the shark"
❌ WRONG: "The visual shows the figure positioned near the shark in what might appear to be an interaction"

Describe what IS visible, not what it "might" or "could" be.

═══════════════════════════════════════════════════════════════════════════════
TASK: Generate {target_questions} adversarial multimodal questions for this frame
═══════════════════════════════════════════════════════════════════════════════

⚠️  CRITICAL: NO HALLUCINATIONS ALLOWED ⚠️

Before generating ANY question, carefully observe ONLY what is actually visible/audible in this frame:

FORBIDDEN HALLUCINATIONS (these will be rejected):
✗ NO inventing interactions that don't exist
  → "gripping", "holding", "using" when objects are just positioned near each other
  → VERIFY: Is there actual physical contact? Actual hand-on-object interaction?

✗ NO inventing states/orientations that aren't visible
  → "overturned", "upside down", "rotated" when object is in normal position
  → VERIFY: Look at the actual orientation carefully

✗ NO inventing details not present in the frame
  → Colors, numbers, text that aren't actually visible
  → Actions, motions, expressions that aren't actually happening

✗ NO assuming relationships or context beyond what's shown
  → "about to", "preparing to", "intending to" (unless motion clearly indicates)
  → VERIFY: What is ACTUALLY happening right now in this frame?

HALLUCINATION EXAMPLES TO AVOID:
✗ BAD: "When crash sound is heard, vehicle is overturned with flames"
  → If vehicle is actually upright, this is a hallucination
✓ GOOD: "When crash sound is heard, flames are visible near the vehicle"

✗ BAD: "When audio occurs, person is gripping the shark's tail"
  → If person and shark are just positioned near each other, this is a hallucination
✓ GOOD: "When audio occurs, person is positioned next to the shark"

✗ BAD: "When sound is heard, person is preparing to throw"
  → If person is just standing, this is a hallucination
✓ GOOD: "When sound is heard, person is standing in ready position"

VERIFICATION CHECKLIST - Before submitting each question:
☐ Is the interaction I'm describing actually happening? (not just spatial proximity)
☐ Is the orientation/state I'm describing what I actually see? (not assumed)
☐ Are the details I'm mentioning actually visible/audible? (not imagined)
☐ Am I describing THIS EXACT MOMENT, not what might happen next?

Your questions must cover these {len(question_types)} types:
"""

    # Add each specialist section
    for i, qtype in enumerate(question_types, 1):
        specialist_prompt = SPECIALIST_PROMPTS.get(qtype, "")

        # ✅ FIXED: Always replace template variables (use fallback if audio_cue is empty)
        # Prevents literal {audio_cue} and {audio_snippet} from appearing in prompts
        audio_cue_value = audio_cue if audio_cue else "[No audio at this timestamp]"
        audio_snippet_value = (audio_cue[:40] + "..." if len(audio_cue) > 40 else audio_cue) if audio_cue else "[No audio]"

        specialist_prompt = specialist_prompt.replace("{audio_cue}", audio_cue_value)
        specialist_prompt = specialist_prompt.replace("{audio_snippet}", audio_snippet_value)

        prompt += f"\n{i}. {qtype.upper()}\n{specialist_prompt}\n"

    # Add output format instructions
    prompt += f"""
═══════════════════════════════════════════════════════════════════════════════
TEMPORAL WINDOW REQUIREMENTS (CRITICAL):
═══════════════════════════════════════════════════════════════════════════════

Your timestamps (start_timestamp and end_timestamp) define the temporal window needed
to answer the question. Use these MINIMUM window sizes based on question type:

SIMPLE OBSERVATION (10-15 second window):
- Needle, Counting, Referential Grounding
- Window: {timestamp:.1f}s ± 6-8 seconds
- Example: Frame at 2:00 → start_timestamp: "01:52", end_timestamp: "02:08" (16s window)

MODERATE COMPLEXITY (20-25 second window):
- Audio-Visual Stitching, Context, General Holistic Reasoning, Inference
- Window: {timestamp:.1f}s ± 10-12 seconds
- Example: Frame at 2:00 → start_timestamp: "01:48", end_timestamp: "02:12" (24s window)

HIGH COMPLEXITY (30-40 second window):
- Sequential, Temporal Understanding, Subscene, Object Interaction Reasoning
- Window: {timestamp:.1f}s ± 15-20 seconds
- Example: Frame at 2:00 → start_timestamp: "01:40", end_timestamp: "02:20" (40s window)

VERY HIGH COMPLEXITY (40-60 second window):
- Comparative, Tackling Spurious Correlations
- Window: {timestamp:.1f}s ± 20-30 seconds
- Example: Frame at 2:00 → start_timestamp: "01:30", end_timestamp: "02:30" (60s window)

⚠️ CRITICAL: Your temporal window MUST be long enough to:
1. Observe the complete action/event
2. See state changes and transitions
3. Track cause-effect relationships
4. Compare before/after states

✗ BAD (too narrow): "02:09" to "02:12" (3 seconds - insufficient for temporal reasoning)
✓ GOOD: "02:00" to "02:30" (30 seconds - sufficient for state changes)

═══════════════════════════════════════════════════════════════════════════════
AUDIO-VISUAL INTEGRATION VALIDATION (CRITICAL):
═══════════════════════════════════════════════════════════════════════════════

⚠️ BEFORE SUBMITTING EACH QUESTION, RUN THESE TESTS:

TEST 1: CAN THIS QUESTION BE ANSWERED USING ONLY THE VISUAL?
  - Read the question and imagine you have NO audio
  - Can you still answer it? → If YES, REJECT THE QUESTION
  - The question MUST require audio to disambiguate or provide context

TEST 2: CAN THIS QUESTION BE ANSWERED USING ONLY THE AUDIO?
  - Read the question and imagine you have NO video
  - Can you still answer it? → If YES, REJECT THE QUESTION
  - The question MUST require visual to answer

TEST 3: DOES THE QUESTION EXPLICITLY LINK AUDIO AND VISUAL?
  - Check if question contains explicit connection words:
    ✓ "when you hear X, what do you see Y?"
    ✓ "how does visual Z relate to audio X?"
    ✓ "what happens visually as audio X occurs?"
  - If the audio and visual are mentioned independently → REJECT

GOOD EXAMPLES (both required):
✓ "When the narrator says 'too slow' in the audio cue, what visual action does
   the figure perform that demonstrates this speed issue?"
   → Requires: Audio (identify "too slow") + Visual (see action) + Connection (demonstrates)

✓ "As the music tempo increases from 80 to 120 BPM, how does the figure's
   movement speed change visually?"
   → Requires: Audio (tempo change) + Visual (movement) + Connection (correlation)

BAD EXAMPLES (single modality sufficient):
✗ "What design is visible on the figure's back?"
   → Can answer with visual only, audio is irrelevant → REJECT

✗ "What does the narrator say about the animation speed?"
   → Can answer with audio only, visual is irrelevant → REJECT

✗ "What color is the object?"
   → Visual only → REJECT

✗ "What sound is heard?"
   → Audio only → REJECT

═══════════════════════════════════════════════════════════════════════════════
CRITICAL REQUIREMENTS - EVERY QUESTION MUST FOLLOW:
═══════════════════════════════════════════════════════════════════════════════

1. Generate EXACTLY {target_questions} questions total
2. Distribute questions across ALL {len(question_types)} types listed above

3. EACH QUESTION MUST:
   ✓ Require BOTH audio AND visual to answer (Rule #1 - CRITICAL!)
   ✓ Cannot be answered from audio alone OR visual alone
   ✓ Specific to this frame (not generic)
   ✓ Clear and unambiguous
   ✓ Follow the requirements for its type

   ═══════════════════════════════════════════════════════════════════════════════
   ⚠️  AUDIO TIMESTAMP VALIDATION (CRITICAL - QUESTIONS WILL BE AUTO-REJECTED)
   ═══════════════════════════════════════════════════════════════════════════════

   Your questions undergo AUTOMATIC VALIDATION. Questions are REJECTED if timestamps point to:

   ❌ SILENT SEGMENTS (RMS energy < 0.01)
      → Audio too quiet to provide meaningful cues
      → Examples: Scene transitions, pauses, black frames, background noise only
      → CHECK: Can you actually HEAR something meaningful at this timestamp?

   ❌ SCENE CUTS (energy change ratio > 3.0)
      → Drastic audio change = scene boundary (different location/context)
      → Examples: Interview → Sports game, Indoor → Outdoor, Quiet → Loud music
      → CHECK: Is audio CONSISTENT across your entire timestamp range?

   ❌ TOO SHORT (duration < 0.1s or temporal window < 10s)
      → Insufficient audio context for meaningful cues
      → CHECK: Is your temporal window at least 10-15 seconds?

   ❌ DUPLICATE AUDIO SEGMENTS (similarity > 85% with other questions)
      → Multiple questions using the same audio clip = redundancy
      → CHECK: Am I using DIFFERENT audio moments than previous questions?

   VALIDATION CHECKLIST - Before submitting EACH question:
   ☐ Can I hear audible speech/sound at BOTH start and end timestamps?
   ☐ Does audio remain consistent (no scene cuts/transitions) across the window?
   ☐ Is my temporal window 10+ seconds long?
   ☐ Am I using a DIFFERENT audio segment than my previous questions?
   ☐ Is this a meaningful audio moment (not silence, not background noise)?

   If you CANNOT verify audio quality → CHOOSE A DIFFERENT TIMESTAMP.

   ═══════════════════════════════════════════════════════════════════════════════

   FORBIDDEN IN QUESTIONS:
   ✗ NO pronouns: he/she/they/him/her/his/their
     → Use: "the person", "the individual", "the person in blue"
   ✗ NO names: no proper names, team names, celebrity names
     → Use: descriptors like "person on left", "individual in uniform"
   ✗ NO timestamp questions: "at what time", "what timestamp"
     → Use: "when you hear X, what do you see?"

   ADVERSARIAL REQUIREMENT (CRITICAL!):
   ✗ REJECT simple observation questions that require no reasoning
     → "What color is X?" = TOO SIMPLE, REJECT
     → "Is the person standing?" = TOO SIMPLE, REJECT
     → "How many objects are visible?" = TOO SIMPLE, REJECT (unless counting requires careful attention)

   ✓ Questions must require CAREFUL ATTENTION to both audio and visual details
   ✓ Avoid questions answerable with a quick glance + listen
   ✓ Require reasoning, integration, or precise detail matching across modalities

   ADVERSARIAL EXAMPLES:
   ✗ BAD (trivial): "What color jersey is worn?"
   ✓ GOOD (requires attention): "What specific details are on the jersey worn by the person speaking during '{{audio_snippet}}'?"

   ✗ BAD (trivial): "Is the person moving?"
   ✓ GOOD (requires reasoning): "What stage of motion is shown when '{{audio_snippet}}' occurs?"

   ✗ BAD (trivial): "What object is visible?"
   ✓ GOOD (requires integration): "How is the visible object being used at the moment '{{audio_snippet}}' is heard?"

4. EACH ANSWER MUST BE RICH AND COMPLETE:

   ANSWER STRUCTURE (MANDATORY 3-PART FORMAT):
   Each answer MUST include these components (50-80 words total):

   1. DIRECT ANSWER (10-15 words)
      - State the answer clearly and specifically
      - No vague or general statements

   2. AUDIO-VISUAL CONNECTION (15-20 words)
      - Explicit connection to the audio cue
      - How audio and visual synchronize or relate
      - Temporal synchronization details

   3. SUPPORTING DETAIL (15-25 words)
      - Most distinctive physical characteristic
      - Specific spatial/temporal context
      - Critical action/interaction detail
      - Additional observable evidence

   ANSWER LENGTH REQUIREMENT:
   ✓ Minimum: 50 words (3-4 sentences)
   ✓ Target: 60-70 words (4 sentences)
   ✗ Too short: Under 45 words = REJECTED (insufficient detail)
   ✗ Too long: Over 85 words = needs editing (too verbose)

   EXAMPLE COMPARISON:

   ❌ BAD (too brief, 8 words):
   "The individual holds a flamethrower producing visible flame."
   → Missing audio connection, spatial context, and supporting details

   ✅ GOOD (rich and complete, 65 words):
   "The individual holds a black and orange flamethrower at waist height, angled
   approximately 45 degrees upward toward the upper left portion of the frame. This
   action occurs when the audio cue 'what about a flamethrower?' is heard, with the
   translucent orange flame element extending 3-4 studs from the nozzle immediately
   following the audio mention. The flame piece uses translucent orange plastic with
   vertical ridges to simulate realistic fire texture."

   ❌ BAD (too brief, 26 words):
   "A large red dragon design is visible on the figure's back when the narrator says
   'animate running', becoming prominently visible as the figure rotates to face the camera."
   → Missing spatial details and supporting physical description

   ✅ GOOD (rich and complete, 72 words):
   "A large red dragon design is visible on the figure's back, occupying the entire
   upper back area from shoulders to waist. This design becomes prominently visible
   when the narrator says 'animate running' at the specific moment, as the figure
   rotates 180 degrees to face the camera. The dragon's wings extend symmetrically
   across both shoulder blades, featuring detailed scales and claws rendered in bright
   red coloring that contrasts sharply with the figure's yellow body."

   ADDITIONAL EXAMPLE (rich detail, 68 words):
   "The person moves from left to right across the frames while the shark remains
   stationary on the right side. This movement occurs as the narrator says 'watch
   the approach' and background music tempo increases from 80 to 120 BPM. The
   person's posture shifts from upright to slightly crouched, suggesting preparation
   for interaction, with the distance between person and shark decreasing from
   approximately 10 studs to 3 studs across the sequence."

   GESTURE INTERPRETATION (ALLOWED):
   When describing gestures or hand movements, you MAY use metaphorical interpretations:
   ✓ "as if counting" → ALLOWED (reasonable interpretation)
   ✓ "like framing a shot" → ALLOWED (clear metaphor)
   ✓ "shaped like a box" → ALLOWED (descriptive comparison)
   ✓ "mimicking weighing something" → ALLOWED (action interpretation)

   IMPORTANT: These interpretations should be REASONABLE and based on visible hand/body positions.
   ✗ DON'T invent complex narratives or backstories
   ✓ DO use common gesture interpretations that match the visual

   VISUAL SCOPE REQUIREMENTS (MANDATORY):
   When describing visual effects, changes, or attributes, you MUST specify the SCOPE:

   Use ONE of these scope indicators:
   - "entire frame" (affects everything visible)
   - "background only" (main subject unaffected)
   - "foreground only" (background unaffected)
   - "specific region: [describe location]" (e.g., "upper left corner", "area behind the person")

   SCOPE EXAMPLES:
   ✗ BAD: "The scene becomes grainy"
      → Unclear: Does this affect the person? Background? Both?
   ✓ GOOD: "The entire frame, including the person and background, displays visible grain"
   ✓ GOOD: "The background behind the person becomes grainy while the person remains in clear focus"

   ✗ BAD: "Red tint is applied"
      → Unclear scope
   ✓ GOOD: "A red tint is applied to the entire frame, affecting both the person and the environment"
   ✓ GOOD: "A red tint affects only the background area while the person remains unaffected"

   FORBIDDEN IN ANSWERS:
   ✗ NO pronouns: he/she/they/him/her → Use "the person", "the individual"
   ✗ NO names or proper nouns
   ✗ NO vague words: something, someone, might, maybe, possibly, seems
   ✗ NO filler words: um, uh, like, basically, literally
   ✗ NO telegraphic answers (under 45 words)
   ✗ NO overly verbose answers (over 85 words)
   ✗ NO answers that just restate the question

5. VISUAL_CUE AND AUDIO_CUE FIELD REQUIREMENTS:
   ✅ FIX #3: The "visual_cue" and "audio_cue" fields (NOT "evidence") must provide OBJECTIVE descriptions

   VISUAL_CUE - Describe what is ACTUALLY VISIBLE in the frame:
   ✓ Specific objects, people, text, colors, positions, actions
   ✓ Precise details (not interpretations): "person in blue jersey with number 7", "hand extended toward basketball"
   ✓ Spatial relationships: "left side", "in foreground", "behind", "holding"
   ✓ Observable actions: "throwing motion", "standing", "fire projecting from device"

   AUDIO_CUE - Describe what is ACTUALLY HEARD:
   ✓ Exact quote if speech: "'pass the ball' is heard"
   ✓ Specific sounds: "click sound", "whooshing sound", "lightsaber clashing"
   ✓ Audio timing: "occurs at this moment", "synchronized with", "aligns with"
   ✓ Audio characteristics: "loud crash", "background music", "rustling"

   FORBIDDEN IN CUES:
   ✗ NO restating the question (cues must be standalone useful)
   ✗ NO interpretations: "looks angry" → use "furrowed eyebrows, gritted teeth"
   ✗ NO pronouns: "he/she" → use "the person", "the individual"
   ✗ NO vague terms: "something", "appears to be", "seems like"

   CUE EXAMPLES:
   ✗ BAD visual_cue: "Jersey visible"
   ✓ GOOD visual_cue: "Blue jersey with white stripes and number 7 worn by person on left"

   ✗ BAD audio_cue: "Sound occurs"
   ✓ GOOD audio_cue: "Phrase 'pass the ball' occurs at this exact moment"

   ✗ BAD visual_cue: "Person looks angry"
   ✓ GOOD visual_cue: "Person displays furrowed eyebrows, red mustache, and gritted mouth"

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (VALID JSON ONLY):
═══════════════════════════════════════════════════════════════════════════════

CRITICAL - EXACT QUESTION TYPE FORMAT:
Use the EXACT type string(s) from your assigned types list above. Match capitalization precisely.

IMPORTANT: Questions can belong to MULTIPLE types. Use an array of types when applicable:
- Single type: ["Referential Grounding"]
- Multiple types: ["Temporal Understanding", "Sequential", "Counting"]
- Multiple types: ["General Holistic Reasoning", "Inference"]

Valid type strings:
- "Needle" (NOT "NEEDLE" or "needle")
- "Audio-Visual Stitching" (NOT "AUDIO-VISUAL STITCHING" or "Audio-Visual")
- "Temporal Understanding" (NOT "TEMPORAL UNDERSTANDING" or "Temporal")
- "Sequential" (NOT "SEQUENTIAL")
- "Subscene" (NOT "SUBSCENE")
- "General Holistic Reasoning" (NOT "GENERAL HOLISTIC REASONING" or "General")
- "Inference" (NOT "INFERENCE")
- "Context" (NOT "CONTEXT")
- "Referential Grounding" (NOT "REFERENTIAL GROUNDING")
- "Counting" (NOT "COUNTING")
- "Comparative" (NOT "COMPARATIVE")
- "Object Interaction Reasoning" (NOT "OBJECT INTERACTION REASONING" or "Object Interaction")
- "Tackling Spurious Correlations" (NOT "TACKLING SPURIOUS CORRELATIONS" or "Spurious")

JSON FORMAT:
{{
  "questions": [
    {{
      "question": "When the audio cue 'pass the ball' is heard, what color jersey is the person on the left wearing?",
      "question_type": ["Referential Grounding"],
      "golden_answer": "The person on the left is wearing a blue jersey with white stripes and number 7 when the phrase 'pass the ball' is heard.",
      "confidence": 0.95,
      "visual_cue": "Blue jersey with white stripes and number 7 worn by person on left side",
      "audio_cue": "Phrase 'pass the ball' occurs at this exact moment",
      "start_timestamp": "02:15",
      "end_timestamp": "02:28"
    }},
    {{
      "question": "What action does the person in blue perform when the audio cue 'pass the ball' is heard?",
      "question_type": ["Audio-Visual Stitching", "Temporal Understanding"],
      "golden_answer": "The person in blue throws the basketball toward a teammate positioned near the basket at the moment 'pass the ball' is heard.",
      "confidence": 0.90,
      "visual_cue": "Throwing motion visible with arm extended, basketball in mid-air toward basket",
      "audio_cue": "Phrase 'pass the ball' synchronized with the throwing action",
      "start_timestamp": "02:15",
      "end_timestamp": "02:35"
    }}
    // ... {target_questions - 2} more questions
  ]
}}

═══════════════════════════════════════════════════════════════════════════════
FINAL CHECK BEFORE SUBMITTING:
═══════════════════════════════════════════════════════════════════════════════

Read each answer aloud. If you hear ANY of these words, REWRITE:
- "appears" → DELETE and replace with "is"
- "seems" → DELETE and replace with "is"
- "looks like" → DELETE and replace with "is"
- "might"/"may"/"could" → DELETE
- "possibly"/"potentially" → DELETE
- "suggesting"/"indicating" → DELETE and replace with direct statement

Example rewrites:
❌ "appears to be a model" → ✅ "is a model"
❌ "might appear to be" → ✅ "is"
❌ "potentially aligning" → ✅ "aligning"
❌ "appear to represent" → ✅ "represent"
❌ "seems to be positioned" → ✅ "is positioned"

VALIDATION CHECKLIST (every question must pass):
☐ Question requires BOTH audio AND visual to answer?
☐ No pronouns (he/she/they) in question or answer?
☐ No names in question or answer?
☐ Answer is 50-80 words (3-4 sentences, 250-400 characters)?
☐ Answer references both audio and visual explicitly?
☐ NO HEDGING WORDS in any answer? (Check: appears/seems/might/possibly/potentially/indicating)
☐ No "at what time" questions?
☐ Proper grammar and capitalization?

IMPORTANT:
- Return ONLY valid JSON, no markdown code blocks
- No ```json``` wrapper
- Generate exactly {target_questions} questions
- Ensure all {len(question_types)} types are represented
- EVERY question must pass all validation checks above
"""

    # ✅ FIXED: Final replacement of any remaining template variables
    # (from f-string sections where {{audio_snippet}} becomes {audio_snippet})
    audio_cue_value = audio_cue if audio_cue else "[No audio at this timestamp]"
    audio_snippet_value = (audio_cue[:40] + "..." if len(audio_cue) > 40 else audio_cue) if audio_cue else "[No audio]"

    prompt = prompt.replace("{audio_cue}", audio_cue_value)
    prompt = prompt.replace("{audio_snippet}", audio_snippet_value)

    return prompt


def get_question_type_priority(question_types: List[str]) -> Dict[str, float]:
    """
    Assign priority scores to question types.
    Higher priority = more important/valuable questions.

    Used to decide which question types get more questions when distributing.

    Returns:
        Dict mapping question_type to priority score (0-1)
    """
    priority_map = {
        # Hard, valuable types (priority: 0.9-1.0)
        "Inference": 1.0,
        "Tackling Spurious Correlations": 0.95,
        "Comparative": 0.92,
        "Subscene": 0.90,

        # Medium difficulty (priority: 0.7-0.8)
        "Audio-Visual Stitching": 0.85,
        "Temporal Understanding": 0.82,
        "Sequential": 0.80,
        "Object Interaction Reasoning": 0.78,

        # Easier but still valuable (priority: 0.5-0.6)
        "General Holistic Reasoning": 0.65,
        "Context": 0.60,
        "Referential Grounding": 0.58,

        # Straightforward (priority: 0.3-0.4)
        "Needle": 0.45,
        "Counting": 0.40,
    }

    return {qt: priority_map.get(qt, 0.5) for qt in question_types}
