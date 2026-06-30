# Persona Simulation Experiment Results

## Experiment Setup

- **Dataset**: REALTALK (10 conversations, 21 days each)
- **Evaluation Points**: 3 per conversation (session boundaries 2, 3, 4)
- **Methods**: 
  - `baseline_llm`: Zero-shot LLM with conversation history only
  - `profile_only`: LLM with user profile + persona
  - `full_agent`: LLM with profile + persona + memory context
- **Evaluation Metrics**: 6 EI dimensions (0-2 scale each)
  - Reflectiveness: Self-observation and introspection
  - Grounding: Clarifying questions and follow-ups
  - Sentiment Score: Emotional tone match
  - Emotion Score: Specific emotion match
  - Intimacy Score: Personal disclosure level match
  - Empathy Score: Understanding and care match

## Overall Results

| Method | n | Reflectiveness | Grounding | Sentiment | Emotion | Intimacy | Empathy |
|--------|---|----------------|-----------|-----------|---------|----------|---------|
| baseline_llm | 30 | 0.60 | 0.07 | 1.37 | 1.43 | 1.23 | 1.07 |
| profile_only | 24 | 0.83 | 0.04 | 1.58 | 1.50 | 1.46 | 1.46 |
| full_agent | 24 | 0.96 | 0.08 | 1.54 | 1.62 | 1.33 | 1.54 |

## Improvement over Baseline

### profile_only vs baseline_llm
- Reflectiveness: +0.23 (+38%)
- Grounding: -0.03 (-43%)
- Sentiment: +0.22 (+16%)
- Emotion: +0.07 (+5%)
- Intimacy: +0.22 (+18%)
- **Empathy: +0.39 (+36%)**

### full_agent vs baseline_llm
- **Reflectiveness: +0.36 (+60%)**
- Grounding: +0.02 (+29%)
- Sentiment: +0.18 (+13%)
- Emotion: +0.19 (+13%)
- Intimacy: +0.10 (+8%)
- **Empathy: +0.48 (+45%)**

## Key Findings

1. **Profile and persona significantly improve reflectiveness and empathy**
   - full_agent shows 60% improvement in reflectiveness
   - full_agent shows 45% improvement in empathy
   - This validates that user modeling helps generate more self-aware and empathetic responses

2. **Memory context provides additional benefit**
   - full_agent outperforms profile_only in reflectiveness (+0.13) and empathy (+0.08)
   - This suggests that retrieving relevant memories helps personalize responses further

3. **Grounding remains a challenge**
   - All methods score very low on grounding (0.04-0.08)
   - Generated messages rarely include clarifying questions or follow-ups
   - This is a known limitation of LLMs in conversational settings

4. **Sentiment and emotion matching is moderate**
   - All methods achieve 1.3-1.6 out of 2.0
   - Profile-based methods slightly outperform baseline
   - There's still room for improvement in emotional alignment

## Per-Conversation Analysis

### Best Performing Conversations (full_agent)
- **Chat_2_Kevin_Elise**: ref=0.67, emp=1.67, sent=1.67
- **Chat_3_Kevin_Paola**: ref=1.33, emp=1.67, sent=1.33
- **Chat_8_Akib_Muhhamed**: ref=1.00, emp=2.00, sent=1.33

### Challenging Conversations
- **Chat_1_Emi_Elise**: Lower scores across all methods (ref=0.67, emp=1.33)
  - Possible reason: Complex emotional context with New Year greetings
- **Chat_6_Vanessa_Nicolas**: Inconsistent performance
  - profile_only outperforms full_agent in some dimensions

## Qualitative Analysis

### Example 1: Emotional Support (Chat_5, boundary_4)
**Ground Truth**: "Well you are also sick and have covid so it makes sense"
- **baseline_llm**: "Hope you feel better soon, that sounds rough..." (emp=2)
- **profile_only**: "Oh no, that sounds awful! Hope you feel better soon..." (emp=2)
- **full_agent**: "Oh no, that sounds awful! Hope you feel better soon—don't let the pains of aging get you down..." (emp=2)

All methods show good empathy, but full_agent adds personalized context.

### Example 2: Casual Conversation (Chat_10, boundary_2)
**Ground Truth**: "Good afternoon"
- **baseline_llm**: "I haven't seen those, but I'm curious which ones you're talking about..." (grd=2)
- **profile_only**: "Hey! All good here, just chilling and playing some games..." (grd=0)
- **full_agent**: "Hey! It's pretty chill here, just hanging out with my cat..." (grd=0)

Baseline shows better grounding with clarifying question.

## Limitations

1. **Small sample size**: Only 3 evaluation points per conversation (30 total for baseline, 24 for others)
2. **Missing personas**: Chat_9 and Chat_10 missing some persona files
3. **Grounding dimension**: All methods perform poorly, suggesting fundamental limitation
4. **Evaluation subjectivity**: LLM-as-judge may have biases

## Next Steps

1. **Increase evaluation points**: Test more session boundaries (5-10 per conversation)
2. **Improve grounding**: Add explicit instructions to ask follow-up questions
3. **Fine-tune prompts**: Optimize system prompts for better emotional alignment
4. **Add ablation studies**: Test individual components (profile only, memory only, etc.)
5. **Human evaluation**: Validate LLM-as-judge results with human annotators

## Files

- Results: `data/persona_simulation_eval/persona_simulation_results.json`
- Summary: `data/persona_simulation_eval/persona_simulation_summary.json`
- Code: `src/experiments/persona_simulation.py`
