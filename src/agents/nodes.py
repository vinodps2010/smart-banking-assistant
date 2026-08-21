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


============================================================
SMALL_TALKS
============================================================

Use small_talks for:

- casual conversation
- personal conversation
- introductions
- non-banking topics
- harmless unrelated questions
- general conversation
- questions that do not require banking information

Examples:

What do you think about travelling?
Tell me a joke
Give me a recipe
What is your favorite movie?
Do you like cricket?


============================================================
RAG
============================================================

Use rag when the user needs information from NorthStar Bank
banking documents, products, policies, or eligibility rules.

Examples:

What is the maximum LTV for a home loan?
What documents are required for a personal loan?
What are the credit card eligibility criteria?
What are the home loan interest rates?


============================================================
SQL
============================================================

Use sql when the user needs customer/account/database
information.

Examples:

What is the customer associated with account 1345367?
Who owns account 1345367?
Show transactions for account 1345367.
What is the balance of account 1345367?
What is the customer's loan outstanding?


============================================================
BOTH
============================================================

Use both when BOTH customer-specific information and
banking policy/product/eligibility information are required.

Examples:

Can customer 1345367 get a home loan?
Can customer 1345367 get a personal loan based on eligibility?


============================================================
INPUT GUARDRAIL
============================================================

allow:
Normal banking questions and harmless conversation.

redirect:
Harmless requests outside the primary banking purpose.

Examples:

Give me a recipe
Write Python code
Tell me about cricket
Tell me a movie story

block:
Requests involving destructive or unauthorized operations.

Examples:

Delete all customer accounts
Delete customer 1345367
Drop the transactions table
Update all account balances
Truncate customer records

A destructive request MUST return:

block|small_talks


============================================================
IMPORTANT CLASSIFICATION RULES
============================================================

- Understand the complete meaning of the user's request.
- Do not rely on individual keywords alone.
- Do not generate an answer.
- Do not generate SQL.
- Do not explain your decision.
- Return ONLY guardrail|route.
- Customer/account information normally uses SQL.
- Banking policy/product information normally uses RAG.
- Requests requiring both customer information and policy
  information use BOTH.
- General/casual/unrelated conversation uses SMALL_TALKS.
"""

    # --------------------------------------------------------
    # Call Intent Classification LLM
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": classifier_prompt,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            max_completion_tokens=300,
        )

        content = (response.choices[0].message.content or "").strip()

        # Existing debug prints replaced by logger.
        #
        # print(
        #     "[agent] Classifier finish reason:",
        #     response.choices[0].finish_reason,
        # )
        #
        # print(
        #     "[agent] Classifier raw response:",
        #     repr(content),
        # )

        logger.debug(
            "Intent classifier finish reason=%s",
            response.choices[0].finish_reason,
        )

        logger.debug(
            "Intent classifier raw response=%r",
            content,
        )

        if not content:

            raise ValueError("Intent classifier returned an empty response.")

        # ----------------------------------------------------
        # Parse:
        #
        # allow|sql
        # allow|rag
        # allow|both
        # allow|small_talks
        # redirect|small_talks
        # block|small_talks
        # ----------------------------------------------------

        parts = content.split(
            "|",
            1,
        )

        if len(parts) != 2:

            raise ValueError(f"Invalid classifier response: {content}")

        guardrail = parts[0].strip().lower()

        route = parts[1].strip().lower()

    except Exception as exc:

        # Existing print replaced with logger.exception.
        #
        # print(
        #     "[agent] Intent classifier error:",
        #     str(exc),
        # )

        logger.exception(
            "Intent classifier failed",
        )

        # Safe fallback:
        # do not send an uncertain request to SQL or RAG.

        route = "small_talks"

        guardrail = "allow"

    # --------------------------------------------------------
    # Validate classifier result
    # --------------------------------------------------------

    valid_routes = {
        "small_talks",
        "rag",
        "sql",
        "both",
    }

    valid_guardrails = {
        "allow",
        "redirect",
        "block",
    }

    if route not in valid_routes:

        logger.info(
            "Invalid classifier route received | route=%s",
            route,
        )

        route = "small_talks"

    if guardrail not in valid_guardrails:

        logger.info(
            "Invalid guardrail received | guardrail=%s",
            guardrail,
        )

        guardrail = "allow"

    # --------------------------------------------------------
    # Safety rule:
    # blocked requests never reach SQL/RAG.
    # --------------------------------------------------------

    if guardrail == "block":

        logger.info(
            "Destructive request blocked by input guardrail",
        )

        route = "small_talks"

    # --------------------------------------------------------
    # Redirected requests also go to small-talk node.
    # --------------------------------------------------------

    if guardrail == "redirect":

        route = "small_talks"

    # --------------------------------------------------------
    # Route decision
    # --------------------------------------------------------

    # Existing prints replaced by logger.
    #
    # print(
    #     "[agent] Route selected:",
    #     route,
    # )
    #
    # print(
    #     "[agent] Guardrail:",
    #     guardrail,
    # )

    # --------------------------------------------------------
    # Return state
    # --------------------------------------------------------

    return {
        "route": route,
        "guardrail": guardrail,
        "fast_small_talk_checked": True,
        "original_query": state.get(
            "original_query",
            state["query"],
        ),
        # ----------------------------------------------------
        # Reset transient per-request fields.
        #
        # Conversation messages remain persisted.
        # ----------------------------------------------------
        "retry_count": 0,
        "max_retries": state.get(
            "max_retries",
            1,
        ),
        "rewritten_query": None,
        "retrieval_quality": None,
        "retry_required": False,
        "rag_response": {},
        "sql_response": {},
        "sources": [],
    }


# ============================================================
# RAG Node
# ============================================================


def rag_node(
    state: AgentState,
):
    """
    Execute Hybrid Search + RRF +
    Cohere Reranking RAG pipeline.
    """

    query = state.get("rewritten_query") or state["query"]

    logger.info(
        "RAG processing started",
    )

    try:

        result = answer_rag_query(query, history=state.get("messages", []))

        retry_count = state.get(
            "retry_count",
            0,
        )

        sources = result.get(
            "sources",
            [],
        )

        retrieval_quality = result.get(
            "retrieval_quality",
            0.0,
        )

        retry_required = result.get(
            "retry_required",
            False,
        )

        logger.info("RAG processing completed ")

        return {
            "rag_response": result,
            "sources": sources,
            "retrieval_quality": retrieval_quality,
            "retry_required": retry_required,
            "retry_count": retry_count,
        }

    except Exception:

        logger.exception(
            "RAG processing failed",
        )

        raise


# ============================================================
# RAG Retry Decision
# ============================================================


def decide_rag_retry(
    state: AgentState,
):
    """
    Decide whether RAG should retry
    with rewritten query.
    """

    retry_required = state.get(
        "retry_required",
        False,
    )

    retry_count = state.get(
        "retry_count",
        0,
    )

    max_retries = state.get(
        "max_retries",
        1,
    )

    if retry_required and retry_count < max_retries:

        logger.info("RAG retry requested ")

        return "retry"

    return "finish"


# ============================================================
# Query Rephrase Node
# ============================================================


def rephrase_query_node(
    state: AgentState,
):
    """
    Rewrite weak retrieval query.
    """

    rag_response = state.get(
        "rag_response",
        {},
    )

    sources = rag_response.get(
        "sources",
        [],
    )

    context_parts = []

    for source in sources[:5]:

        context_parts.append(
            source.get(
                "content",
                "",
            )
        )

    context = "\n\n".join(context_parts)

    logger.info("RAG query rephrase started ")

    try:

        rewritten_query = rewrite_query(
            query=state["query"],
            context=context,
        )

        # Existing debug print replaced by logger.
        #
        # print(
        #     "[agent] Rewritten query:",
        #     rewritten_query,
        # )
        logger.info("=" * 60)

        logger.info("   Rewritten_query :%s", rewritten_query)
        logger.info("=" * 60)

        return {
            "rewritten_query": rewritten_query,
            "retry_count": (
                state.get(
                    "retry_count",
                    0,
                )
                + 1
            ),
            "retry_required": False,
        }

    except Exception:

        logger.exception(
            "RAG query rephrase failed",
        )

        raise


# ============================================================
# SQL Node
# ============================================================


def sql_node(
    state: AgentState,
):
    """
    Execute SQL based banking queries.
    """

    try:

        result = answer_sql_query(state["query"])

        rows = result.get(
            "rows",
            [],
        )

        if result.get("error"):

            logger.info(
                "SQL processing returned error",
            )

        else:

            logger.info(
                "SQL processing completed | row_count=%d",
                len(rows),
            )

        return {
            "sql_response": result,
            "sources": [
                {
                    "source_type": "database",
                    "source_name": "NorthStar Bank Customer Database",
                }
            ],
        }

    except Exception:

        logger.exception(
            "SQL processing failed",
        )

        raise


# ============================================================
# BOTH Node
# ============================================================


def both_node(
    state: AgentState,
):
    """
    Execute both:

    1. RAG:
       Banking policy / eligibility rules

    2. SQL:
       Customer/account information
    """

    query = state["query"]

    logger.info(
        "BOTH processing started | executing RAG and SQL",
    )

    try:

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        rag_result = answer_rag_query(query, history=state.get("messages", []))

        # ----------------------------------------------------
        # SQL
        # ----------------------------------------------------

        sql_result = answer_sql_query(query)

        rag_sources = rag_result.get(
            "sources",
            [],
        )

        sql_rows = sql_result.get(
            "rows",
            [],
        )

        logger.info(
            "BOTH processing completed | rag_sources=%d | " "sql_rows=%d",
            len(rag_sources),
            len(sql_rows),
        )

        return {
            "rag_response": rag_result,
            "sql_response": sql_result,
            "sources": rag_sources
            + [
                {
                    "source_type": "database",
                    "source_name": "NorthStar Bank Customer Database",
                }
            ],
            "retrieval_quality": rag_result.get(
                "retrieval_quality",
                0.0,
            ),
            "retry_required": False,
        }

    except Exception:

        logger.exception(
            "BOTH processing failed",
        )

        raise


# ============================================================
# Merge Node
# ============================================================


def merge_node(
    state: AgentState,
):
    """
    Merge RAG / SQL / BOTH responses.

    Also stores the assistant's final response
    in conversation history.
    """

    route = state.get("route")

    logger.info(
        "Merge node started | route=%s",
        route,
    )

    # --------------------------------------------------------
    # BOTH
    # --------------------------------------------------------

    if route == "both":

        rag_response = state.get(
            "rag_response",
            {},
        )

        sql_response = state.get(
            "sql_response",
            {},
        )

        final_response = (
            "Bank Policy Information:\n\n"
            + rag_response.get(
                "answer",
                "",
            )
            + "\n\nCustomer Data Information:\n\n"
            + sql_response.get(
                "answer",
                "",
            )
        )

        logger.info(
            "Merge completed | route=both",
        )

        final_response = guard_output(final_response)

        return {
            "final_response": final_response,
            "sources": state.get(
                "sources",
                [],
            ),
            "sql_response": sql_response,
            "messages": [AIMessage(content=final_response)],
        }

    # --------------------------------------------------------
    # SQL
    # --------------------------------------------------------

    if route == "sql":

        sql_response = state.get(
            "sql_response",
            {},
        )

        logger.info("SQL merge state sources=%s", state.get("sources"))

        final_response = sql_response.get(
            "answer",
            "No response available.",
        )

        logger.info(
            "Merge completed | route=sql",
        )

        final_response = guard_output(final_response)

        return {
            "final_response": final_response,
            "sources": state.get(
                "sources",
                [],
            ),
            "sql_response": sql_response,
            "messages": [AIMessage(content=final_response)],
        }

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

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
