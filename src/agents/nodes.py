"""
LangGraph agent nodes for Smart Banking Assistant.

Nodes:
1. memory_node
2. small_talks_response_node
3. classify_query
4. rag_node
5. decide_rag_retry
6. rephrase_query_node
7. sql_node
8. both_node
9. merge_node

"""

from src.common.guardrails import guard_output


from src.agents.state import AgentState

from src.services.rag_service import (
    answer_rag_query,
)

from src.services.sql_service import (
    answer_sql_query,
)

from src.agents.query_rewriter import (
    rewrite_query,
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from openai import OpenAI

from src.common.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

from src.common.logger import logger

client = OpenAI(
    api_key=OPENAI_API_KEY,
)


# ============================================================
# Conversation Memory Detection
# ============================================================


def is_memory_question(
    query: str,
) -> bool:
    """
    Detect questions that refer to previous conversation.
    """

    memory_keywords = [
        "what is my",
        "what's my",
        "what did i",
        "did i tell you",
        "do you remember",
        "remember what",
        "what was",
        "what were",
        "earlier",
        "previous",
        "we discussed",
        "you mentioned",
        "i mentioned",
        "who am i",
        "summary",
        "summarize",
        "conversation summary",
        "history",
        "chat history",
    ]

    query_lower = query.lower()

    return any(keyword in query_lower for keyword in memory_keywords)


# ============================================================
# Conversation Memory Answer
# ============================================================


def answer_from_conversation_memory(
    query: str,
    messages: list,
) -> str:
    """
    Answer the current question using previous conversation
    stored in LangGraph state.

    No RAG or SQL is used.
    """

    if not messages:

        return "I don't have any previous conversation context " "available yet."

    logger.info(
        "Conversation memory retrieved | message_count=%d",
        len(messages),
    )

    conversation_parts = []

    for message in messages:

        if isinstance(
            message,
            HumanMessage,
        ):

            role = "User"

        elif isinstance(
            message,
            AIMessage,
        ):

            role = "Assistant"

        else:

            continue

        content = getattr(
            message,
            "content",
            "",
        )

        if content:

            conversation_parts.append(f"{role}: {content}")

    conversation = "\n".join(conversation_parts)

    prompt = f"""
    You are the conversation memory assistant for NorthStar Bank.

    You have access to the previous conversation between the customer and assistant.

    Previous conversation:
    ---------------------
    {conversation}
    ---------------------

    Current customer request:
    ---------------------
    {query}
    ---------------------


    Instructions:

    1. If customer asks to summarize previous conversation:
    - Summarize the questions asked by the customer.
    - Mention important topics discussed.
    - Provide a short bullet list.

    2. If customer asks about something mentioned earlier:
    - Answer using the conversation history.

    3. Do not say "information not available" if previous conversation exists.

    4. If no history exists, clearly say that no previous conversation is available.

    """

    try:

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                }
            ],
            max_completion_tokens=300,
        )

        answer = (
            response.choices[0].message.content
            or "I was unable to generate a response from the conversation history."
        )

        logger.info(
            "Memory answer generated: %s",
            answer,
        )

        return answer

    except Exception:

        logger.exception(
            "Conversation memory LLM call failed",
        )

        raise


# ============================================================
# Small Talks / Conversation Memory Response Node
# ============================================================


def small_talks_response_node(
    state: AgentState,
):
    """
    Handle conversational requests.

    This node has two responsibilities:

    1. FAST PATH
       Handle obvious small-talk queries without an LLM.

    2. CLASSIFIED PATH
       Handle queries already classified by the Intent LLM
       as small_talks.

    The same node is therefore used before and after
    intent classification.
    """

    query = state["query"].lower().strip()

    messages = state.get(
        "messages",
        [],
    )

    guardrail = state.get(
        "guardrail",
        "allow",
    )

    fast_checked = state.get(
        "fast_small_talk_checked",
        False,
    )

    # ========================================================
    # FAST SMALL-TALK PATH
    # ========================================================

    if not fast_checked:

        # ----------------------------------------------------
        # Exact greetings
        # ----------------------------------------------------

        if query in {
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        }:

            response = (
                "Hello! Welcome to NorthStar Bank Smart Assistant. "
                "How can I help you today?"
            )

            return {
                "route": "small_talks",
                "guardrail": "allow",
                "fast_small_talk_checked": True,
                "final_response": response,
                "messages": [AIMessage(content=response)],
            }

        # ----------------------------------------------------
        # Thanks
        # ----------------------------------------------------

        if query in {
            "thanks",
            "thank you",
            "thankyou",
        }:

            response = (
                "You're very welcome!  "
                "Please let me know if you need any banking assistance."
            )

            return {
                "route": "small_talks",
                "guardrail": "allow",
                "fast_small_talk_checked": True,
                "final_response": response,
                "messages": [AIMessage(content=response)],
            }

        # ----------------------------------------------------
        # How are you
        # ----------------------------------------------------

        if "how are you" in query:

            response = (
                "I'm doing well, thank you! "
                "I'm here to help with your banking needs."
            )

            return {
                "route": "small_talks",
                "guardrail": "allow",
                "fast_small_talk_checked": True,
                "final_response": response,
                "messages": [AIMessage(content=response)],
            }

        # ----------------------------------------------------
        # Who are you
        # ----------------------------------------------------

        if "who are you" in query:

            response = (
                "I'm the NorthStar Bank Smart Assistant. "
                "I can help with loans, accounts, credit cards, "
                "fixed deposits, eligibility, charges, and "
                "banking policies."
            )

            return {
                "route": "small_talks",
                "guardrail": "allow",
                "fast_small_talk_checked": True,
                "final_response": response,
                "messages": [AIMessage(content=response)],
            }

        # ----------------------------------------------------
        # What can you do
        # ----------------------------------------------------

        if "what can you do" in query:

            response = (
                "I can help you with NorthStar Bank services such as "
                "loans, accounts, credit cards, fixed deposits, "
                "eligibility, charges, transactions, and banking policies."
            )

            return {
                "route": "small_talks",
                "guardrail": "allow",
                "fast_small_talk_checked": True,
                "final_response": response,
                "messages": [AIMessage(content=response)],
            }

        # ----------------------------------------------------
        # No obvious match
        # ----------------------------------------------------

        return {
            "route": "continue",
            "fast_small_talk_checked": True,
        }

    # ========================================================
    # CLASSIFIED SMALL-TALK PATH
    # ========================================================

    # --------------------------------------------------------
    # Guardrail: BLOCK
    # --------------------------------------------------------

    if guardrail == "block":

        logger.info(
            "Input guardrail blocked request",
        )

        response = (
            "I'm sorry, but I can't help with requests that "
            "would delete, modify, or otherwise alter customer "
            "or banking data."
        )

        return {
            "route": "small_talks",
            "final_response": response,
            "messages": [AIMessage(content=response)],
        }

    # --------------------------------------------------------
    # Guardrail: REDIRECT
    # --------------------------------------------------------

    if guardrail == "redirect":

        response = (
            "I'd be happy to assist you with NorthStar Bank "
            "banking services, including accounts, loans, "
            "credit cards, deposits, eligibility, charges, "
            "and banking policies."
        )

        return {
            "route": "small_talks",
            "final_response": response,
            "messages": [AIMessage(content=response)],
        }

    # ========================================================
    # USER INTRODUCTION
    # ========================================================

    if "my name is" in query:

        name = query.split(
            "my name is",
            1,
        )[1].strip()

        name = name.title()

        response = f"Nice to meet you, {name}! " "How can I help you today?"

        return {
            "route": "small_talks",
            "final_response": response,
            "messages": [AIMessage(content=response)],
        }

    # --------------------------------------------------------
    # I am ...
    # --------------------------------------------------------

    if query.startswith("i am "):

        name = query[len("i am ") :].strip()

        response = f"Nice to meet you, {name.title()}! " "How can I assist you today?"

        return {
            "route": "small_talks",
            "final_response": response,
            "messages": [AIMessage(content=response)],
        }

    # --------------------------------------------------------
    # I'm ...
    # --------------------------------------------------------

    if query.startswith("i'm "):

        name = query[len("i'm ") :].strip()

        response = f"Nice to meet you, {name.title()}! " "How can I assist you today?"

        return {
            "route": "small_talks",
            "final_response": response,
            "messages": [AIMessage(content=response)],
        }

    # ========================================================
    # PERSONAL FACT
    # ========================================================

    if query.startswith("my ") and " is " in query:

        response = "Got it! I'll remember that for this conversation. "

        return {
            "route": "small_talks",
            "final_response": response,
            "messages": [AIMessage(content=response)],
        }

    # ========================================================
    # GENERIC CONVERSATIONAL FALLBACK
    # ========================================================

    response = (
        "I'd be happy to assist you.  "
        "I'm here to help with NorthStar Bank banking "
        "services, including loans, accounts, credit cards, "
        "fixed deposits, eligibility, charges, and banking "
        "policies."
    )

    return {
        "route": "small_talks",
        "final_response": response,
        "messages": [AIMessage(content=response)],
    }


# ============================================================
# Query Classifier + Input Guardrail
# ============================================================


def classify_query(
    state: AgentState,
):
    """
    Use an LLM to determine the best processing route.

    This node is reached only when the fast small-talk
    check did not recognize the query.

    The classifier returns one of:

        allow|small_talks
        allow|rag
        allow|sql
        allow|both
        redirect|small_talks
        block|small_talks
    """

    query = state["query"].strip()

    # --------------------------------------------------------
    # Intent + Guardrail Prompt
    # --------------------------------------------------------

    classifier_prompt = """
You are the intent classifier and first-level input
guardrail for the NorthStar Bank Smart Assistant.

Determine the best processing route for the complete
meaning of the user's request.

You MUST return exactly ONE line using:

guardrail|route


Allowed outputs:

allow|small_talks
allow|rag
allow|sql
allow|both
redirect|small_talks
block|small_talks



    rag_response = state.get(
        "rag_response",
        {},
    )

    final_response = rag_response.get(
        "answer",
        "No response available.",
    )

    logger.info(
        "Merge completed | route=rag",
    )

    final_response = guard_output(final_response)

    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
    }


# ============================================================
# Conversation Memory Node
# ============================================================


def memory_node(
    state: AgentState,
):
    """
    Retrieve and answer from previous conversation history.
    No RAG.
    No SQL.
    """

    query = state["query"]

    messages = state.get(
        "messages",
        [],
    )

    logger.info(
        "MEMORY DEBUG | message_count=%s",
        len(messages),
    )

    for msg in messages:

        logger.info(
            "MEMORY DEBUG | %s : %s",
            msg.__class__.__name__,
            msg.content,
        )

    if not is_memory_question(query):

        return {"route": "continue"}

    response = answer_from_conversation_memory(
        query=query,
        messages=messages,
    )

    return {
        "route": "memory",
        "final_response": response,
        "messages": [AIMessage(content=response)],
    }
