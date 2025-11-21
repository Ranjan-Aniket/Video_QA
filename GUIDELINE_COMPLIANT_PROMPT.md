# Guideline-Compliant Question Generation Prompt

## CRITICAL GUIDELINES (Must Follow ALL)

### 1. DUAL CUE REQUIREMENT
- EVERY question MUST require BOTH audio AND visual information
- If answerable with just one cue → REJECT immediately
- Example BAD: "What color is the lighthouse when X is said?" (If only 1 lighthouse → visual only)
- Example GOOD: "What color is the lighthouse when you hear the second bell chime?" (Requires counting audio cues + visual)

### 2. NO NAMES - EVER
- NEVER use: person names, team names, company names, movie/song/book titles
- ALWAYS use descriptors: "player in white #10", "man with red hat", "lead female character"
- NEVER use: he/she/him/her/they/them/their
- ALWAYS use: "the player", "the individual wearing orange"

### 3. AUDIO CUE DIVERSITY
Audio cues should include:
- Speech/dialogue (with exact quotes)
- Background music (piano playing, tempo change, sudden stop)
- Environmental sounds (birds chirping, door closing, glass breaking)
- Tone/pitch changes (music gets louder, voice deepens)
- Audience reactions (clapping, gasping, cheering)
- Sound effects (whoosh, beep, alarm)

### 4. QUESTION PATTERN VARIETY
DO NOT repeat "When you hear X, what do you see?"

GOOD PATTERNS:
- "What happens before [audio cue]?"
- "What happens after [audio cue]?"
- "When you see [visual], what do you hear?"
- "What is [person] doing when [audio]?"
- "What prompts [sound] to occur?"
- "What visual elements appear as [audio] starts?"
- "How many [objects] are visible when [audio]?"
- "What is the difference in [visual] before and after [audio]?"
- "What order: (A) [visual] (B) [audio] (C) [visual]?"

### 5. TASK TYPE GUIDELINES

**Temporal Understanding** - before/after/when
- "What happens before the individual says 'X'?"

**Sequential** - order of events
- "What is the order: (A) person waves (B) music starts (C) door opens?"
- "Which happens first: the whistle or the player jumping?"

**Inference** - Why/purpose/meaning (MUST ask "why" or "purpose")
- "Why does the person in blue yell after lifting the grill handle?"
- "What is the purpose of the musical jingle at the start?"

**Context** - Background/foreground/setting elements
- "What visual elements are in the background when the person says 'X'?"
- "What billboard text appears when [audio]?"

**Needle** - Specific details (text, graphics, specific objects)
- "What text pops up when the person says 'check the link'?"
- "What jersey number is visible when the announcer says 'three-pointer'?"

**Referential Grounding** - Connect audio and visual at specific moment
- "What is the player doing with the ball when you hear 'great move'?"
- "Who is visible when the person says 'protocol with Morgan'?"

**Counting** - Count specific elements
- "How many times does the coach stand up after hearing a whistle?"
- "How many plates are stacked before the person says 'watch this'?"

**Comparative** - Before/after differences
- "What is the difference in the person's shirt before and after saying 'appreciate it'?"
- "What changes in the scene after the music tempo doubles?"

**Object Interaction** - How objects change through actions
- "How does the clay change after the person discusses 'pour over'?"
- "What effect occurs when the individual places the prism?"

**Subscene** - Caption a segment with both audio and visual
- "Describe what happens when the score shows 118-2"
- "What happens during the warehouse fight scene when the narrator says 'shadows'?"

**Spurious Correlations** - Unexpected/unintuitive events
- "Who are they referring to when saying 'charging bull'?" (Answer: Superman hologram)
- "What unique event occurs with the cone-shaped cylinder?"

**Audio-Visual Stitching** - Editing choices, clip transitions
- "Is the inventor in the same room or a spliced clip?"
- "How do the musical clips pace the interview?"

**Holistic Reasoning** - Entire video knowledge required
- "What is the purpose of breaking the game into clips?"
- "How do the falling bars relate to the song?"

### 6. ANSWER REQUIREMENTS
- 1-2 sentences maximum
- Specific and concise
- NO description dumps
- NO "Based on the visual analysis..."
- GOOD: "3 plates, 2 bowls, and 1 cup"
- BAD: "When analyzing the visual cues provided, we can observe that..."

### 7. TIMESTAMP ACCURACY
- NO questions like "At what time is X said?"
- Timestamps must cover EXACTLY the cues and actions mentioned
- Start: First cue appears
- End: Last action/quote completes

### 8. PRECISION
- Quotes must be transcribed EXACTLY
- Visual descriptions must be ACCURATE (don't say "blue shirt" if it's black)
- No ambiguity allowed

## EXAMPLE PROMPT TEMPLATE

```
Generate a {TASK_TYPE} multimodal video question.

AUDIO CUE TYPE: {speech|music|environmental_sound|tone_change|sound_effect}
AUDIO CUE: {exact audio description}
TIMESTAMP: {timestamp}s

VISUAL CONTEXT: {gpt4v_description}

TASK TYPE: {TASK_TYPE}
SPECIFIC REQUIREMENT: {task-specific instruction}

CRITICAL RULES:
1. Question MUST require BOTH audio AND visual to answer
2. NO NAMES (people, teams, companies, movies, songs) - use descriptors ONLY
3. NO PRONOUNS (he/she/him/her/they/them) - use "the player", "the individual in red"
4. Use diverse question patterns (not just "when you hear X what do you see")
5. Answer must be 1-2 sentences, specific, NO description dumps
6. Transcribe quotes EXACTLY
7. Visual details must be ACCURATE

OUTPUT FORMAT:
Question: {question following pattern for this task type}
Answer: {1-2 sentence specific answer}
Audio Cue: {exact audio reference}
Visual Cue: {concise visual element description}

EXAMPLES FOR THIS TASK TYPE:
{provide 2-3 examples specific to the task type}
```

## VALIDATION CHECKLIST
Before accepting a question, verify:
- [ ] Has BOTH audio and visual cues?
- [ ] Cannot be answered with just one cue?
- [ ] NO names or pronouns?
- [ ] NOT a timestamp question?
- [ ] Answer is 1-2 sentences, specific?
- [ ] Follows the task type definition correctly?
- [ ] Audio cue is diverse (not just speech)?
- [ ] Question pattern is varied (not repetitive)?
