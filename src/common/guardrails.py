import os
import re
import json


from dotenv import load_dotenv

from guardrails import Guard


from guardrails.hub import (
    GuardrailsPII,
    ToxicLanguage,
)

from guardrails.errors import ValidationError

from src.common.logger import logger


import logging

logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("gliner").setLevel(logging.ERROR)

load_dotenv()


# =====================================================
# Configuration
# =====================================================


PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "PERSON",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "EMAIL",
    "CUSTOMER_NAME",
    "CARD_ID",
]


CUSTOMER_ID_PATTERN = re.compile(r"\b\d{6,}\b")


TOXICITY_THRESHOLD = 0.5


def configure_guardrail_logging():

    noisy_loggers = [
        "presidio-analyzer",
        "huggingface_hub",
        "gliner",
    ]

    for logger_name in noisy_loggers:

        logging.getLogger(logger_name).setLevel(logging.ERROR)


configure_guardrail_logging()


# =====================================================
# Build Guards
# =====================================================


def build_input_guard():

    return Guard().use(
        ToxicLanguage(
            threshold=TOXICITY_THRESHOLD,
            validation_method="sentence",
            on_fail="exception",
        )
    )


def build_output_guard():

    return Guard().use(
        GuardrailsPII(
            entities=PII_ENTITIES,
            on_fail="fix",
        )
    )


_guards = None


def get_guards():

    global _guards

    if _guards is None:

        _guards = {
            "input": build_input_guard(),
            "output": build_output_guard(),
        }

    return _guards


def guard_input(query: str):

    guards = get_guards()

    try:

        guards["input"].validate(query)

        logger.info("Input guard passed")

    except ValidationError:

        logger.warning("Input blocked by toxicity guard")

        raise ValueError("Your message contains inappropriate language.")


def guard_output(response: str):

    if not response:
        return response

    """
    response = CUSTOMER_ID_PATTERN.sub(
        "<ACCOUNT_ID>",
        response,
    )
    """

    guards = get_guards()

    try:

        result = guards["output"].validate(response)

        # logger.info("Output PII guard applied")

        return getattr(result, "validated_output", None) or response

    except Exception:

        logger.exception("PII masking failed")

        return response


def guard_sql_result(data):

    if not data:
        return data

    try:

        masked_rows = []

        for row in data:

            row = mask_customer_fields(row)

            json_text = json.dumps(row, default=str)

            masked_text = guard_output(json_text)

            masked_rows.append(json.loads(masked_text))

        return masked_rows

    except Exception:

        logger.exception("SQL PII masking failed")

        return data


def mask_customer_fields(row):

    sensitive_fields = [
        "customer_name",
        "name",
    ]

    for field in sensitive_fields:

        if field in row and row[field]:

            row[field] = "<CUSTOMER_NAME>"

    return row
