import re
from typing import Dict, List, Optional
from datasets import load_dataset

QUERY_TEMPLATE_RU = """
Реши следующую математическую задачу пошагово. Последняя строка твоего ответа должна быть в формате Ответ: $ANSWER (без кавычек, скобок и текстового форматирования), где $ANSWER - это ответ на задачу. Ответ должен быть точным, если необходимо - несократимой дробью через точку. Если задача подразумевает перечисления - выпиши все ответы слитно без разделителей. Если в задаче требуется найти несколько неизвестных - перечисляй их через ";". Используй те единицы измерения, которые содержатся в условии, если в нем не сказано обратного, сами единицы измерения в ответ не записывай. Если требуется выбрать что-то перечислительное - напиши только само число. После ответа не пиши ничего. Далее сама задача:

{task}

Не забудь написать ответ в отдельной строке после "Ответ:", без использования команды \\boxed и в нужном формате.
""".strip()

PHYSICS_TEMPLATE_RU = """
Реши следующую задачу по физике пошагово. Последняя строка твоего ответа должна быть в формате Ответ: $ANSWER (без кавычек, скобок и текстового форматирования), где $ANSWER - это ответ на задачу. Ответ должен быть точным, если необходимо - несократимой дробью через точку. Если задача подразумевает перечисления - выпиши все ответы слитно без разделителей. Если в задаче требуется найти несколько неизвестных - перечисляй их через ";". Используй те единицы измерения, которые содержатся в условии, если в нем не сказано обратного, сами единицы измерения в ответ не записывай. После ответа не пиши ничего. Если требуется выбрать что-то перечислительное - напиши только само число. После ответа не пиши ничего. Далее сама задача:

{task}

Не забудь написать ответ в отдельной строке после "Ответ:", без использования команды \\boxed и в нужном формате.
""".strip()


class RussianMathEval():
    """
    Класс для оценки языковых моделей на русскоязычных математических задачах.
    """

    def __init__(
        self,
        equality_checker, 
        num_examples: Optional[int] = None,
        n_repeats: int = 1,
        debug: bool = False,
    ) -> None:
        """
        Инициализирует оценку на русскоязычных математических задачах.

        Args:
            equality_checker: Объект для проверки равенства ответов
            num_examples: Количество примеров для оценки (по умолчанию 5)
            n_repeats: Количество повторений набора примеров
            debug: Режим отладки для подробного вывода
        """
        dataset = load_dataset("Vikhrmodels/russian_math")
        examples = [
            {"task": row["task"], "Answer": row["short answer"]}
            for row in dataset["train"]
        ]

        if num_examples and num_examples > 0:
            examples = examples[:num_examples]

        self.examples: List[Dict[str, str]] = examples * n_repeats
        self.equality_checker = equality_checker
        self.debug: bool = debug

        if self.debug:
            print(f"Loaded {len(self.examples)} examples for evaluation")

    def __call__(self, sampler, batch_size: int = 1) -> float:
        """
        Выполняет оценку модели на математических задачах.

        Args:
            sampler: Модель для оценки

        Returns:
            Результат оценки модели
        """
        score: float = 0
        for row in self.examples:
            if self.debug:
                    print("\nDebug: Processing example")
                    print(f"Task: {row['task']}")
                    print(f"Expected answer: {row['Answer']}")

            prompt_messages = {"role": "user", "content": QUERY_TEMPLATE_RU.format(**row)}

            response_text = sampler(prompt_messages)

            answer_pattern = r"(?:Answer|Ответ):\s*(.+)$"
            matches = list(re.finditer(answer_pattern, response_text, re.MULTILINE))
            extracted_answer = matches[-1].group(1).strip() if matches else None

            if self.debug:
                print(f"Extracted answer: {extracted_answer}")

            score += float(
                    self.equality_checker(str(row["Answer"]), extracted_answer)
            )

            if self.debug:
                print(f"Score: {score}")

        return score / len(self.examples)


class RussianPhysicsEval():
    """
    Класс для оценки языковых моделей на русскоязычных задачах по физике.
    """

    def __init__(
        self,
        equality_checker,
        num_examples: Optional[int] = None,
        n_repeats: int = 1,
        debug: bool = False,
    ) -> None:
        """
        Инициализирует оценку на русскоязычных задачах по физике.

        Args:
            equality_checker: Объект для проверки равенства ответов
            num_examples: Количество примеров для оценки (по умолчанию 5)
            n_repeats: Количество повторений набора примеров
            debug: Режим отладки для подробного вывода
        """
        dataset = load_dataset("Vikhrmodels/russian_physics")
        examples = [
            {"task": row["task"], "Answer": row["answer"]} for row in dataset["train"]
        ]

        if num_examples and num_examples > 0:
            examples = examples[:num_examples]

        self.examples: List[Dict[str, str]] = examples * n_repeats
        self.equality_checker = equality_checker
        self.debug: bool = debug

        if self.debug:
            print(f"Loaded {len(self.examples)} physics examples for evaluation")

    def __call__(self, sampler) -> float:
        """
        Выполняет оценку модели на задачах по физике.

        Args:
            sampler: Модель для оценки

        Returns:
            Результат оценки модели
        """

        score: float = 0
        for row in self.examples:
            if self.debug:
                    print("\nDebug: Processing example")
                    print(f"Task: {row['task']}")
                    print(f"Expected answer: {row['Answer']}")

            prompt_messages = {"role": "user", "content": QUERY_TEMPLATE_RU.format(**row)}

            response_text = sampler(prompt_messages)

            answer_pattern = r"(?:Answer|Ответ):\s*(.+)$"
            matches = list(re.finditer(answer_pattern, response_text, re.MULTILINE))
            extracted_answer = matches[-1].group(1).strip() if matches else None

            if self.debug:
                print(f"Extracted answer: {extracted_answer}")

            score += float(
                    self.equality_checker(str(row["Answer"]), extracted_answer)
            )

            if self.debug:
                print(f"Score: {score}")

        return score / len(self.examples)