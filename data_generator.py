"""
STM32Cube Synthetic Data Generator
Produces JSON documents that mirror real STM32Cube knowledge base structure.
Replace with your actual JSON files for production benchmarking.
"""

import json
import random
from pathlib import Path

# ── STM32 domain content pools ─────────────────────────────────────────────

PERIPHERALS = ["UART", "SPI", "I2C", "TIM", "ADC", "DAC", "DMA", "GPIO",
                "RCC", "WWDG", "IWDG", "USB", "CAN", "SDMMC", "LTDC", "RNG"]

MCU_FAMILIES = ["STM32F4", "STM32H7", "STM32L4", "STM32G0", "STM32U5",
                 "STM32F7", "STM32WB", "STM32MP1"]

TOPICS = {
    "init":      "Initialization and configuration",
    "irq":       "Interrupt handling and NVIC configuration",
    "dma":       "DMA transfer setup and linked-list mode",
    "clock":     "Clock tree configuration and PLL setup",
    "power":     "Low-power modes and wake-up sources",
    "debug":     "Debug interface, SWD, and trace",
    "security":  "TrustZone and secure boot",
    "rtos":      "FreeRTOS integration and task scheduling",
}

PARAGRAPH_TEMPLATES = [
    (
        "To initialize the {periph} peripheral on {mcu}, call HAL_{periph}_Init() after "
        "configuring the HAL_{periph}_MspInit() callback. The MSP init must enable the "
        "peripheral clock via __HAL_{periph}_CLK_ENABLE() and configure the associated "
        "GPIO pins using HAL_GPIO_Init() with the correct alternate function."
    ),
    (
        "The {periph} on {mcu} supports DMA-based transfers. Configure the DMA stream "
        "by calling HAL_DMA_Init() with the correct channel, direction (memory-to-peripheral "
        "or peripheral-to-memory), data width, and burst size. Link the DMA handle to the "
        "{periph} handle via __HAL_LINKDMA() before starting any transfer."
    ),
    (
        "Interrupt-driven {periph} communication on {mcu} requires enabling the global "
        "interrupt in NVIC using HAL_NVIC_EnableIRQ({periph}_IRQn) with an appropriate "
        "priority. Implement the {periph}_IRQHandler() and call the corresponding "
        "HAL_{periph}_IRQHandler() dispatcher inside it."
    ),
    (
        "Clock configuration for {periph} on {mcu} is managed through the RCC subsystem. "
        "Use HAL_RCC_OscConfig() to configure the PLL source and multipliers, then call "
        "HAL_RCC_ClockConfig() to select HCLK, PCLK1, and PCLK2 dividers. Ensure the "
        "{periph} clock does not exceed the maximum frequency specified in the datasheet."
    ),
    (
        "When using {periph} in low-power mode on {mcu}, select the appropriate LPUART or "
        "LP variant if available. Configure the autonomous mode register to allow the "
        "peripheral to run while the CPU is in Stop mode. Use the LPTIM as a timebase "
        "instead of SysTick when the system clock is gated."
    ),
    (
        "The STM32CubeMX code generator produces a MX_{periph}_Init() function that "
        "encapsulates all handle initialization for {periph} on {mcu}. This function should "
        "be called from the main() initialization sequence before the application loop. "
        "Avoid re-initializing handles at runtime unless a full peripheral reset is required."
    ),
    (
        "Error handling for {periph} on {mcu} relies on HAL status return codes: HAL_OK, "
        "HAL_ERROR, HAL_BUSY, and HAL_TIMEOUT. Always check the return value of blocking "
        "calls such as HAL_{periph}_Transmit() and implement a timeout recovery strategy "
        "to prevent deadlocks in production firmware."
    ),
    (
        "For {periph} on {mcu}, the HAL callback mechanism allows non-blocking operation. "
        "Register callbacks using HAL_{periph}_RegisterCallback() when USE_HAL_XXX_REGISTER_CALLBACKS "
        "is enabled in stm32xx_hal_conf.h. Default weak callbacks (HAL_{periph}_TxCpltCallback, "
        "HAL_{periph}_ErrorCallback) can be overridden directly in user code."
    ),
]

FAQ_TEMPLATES = [
    "What is the correct initialization sequence for {periph} on {mcu}?",
    "How do I configure DMA for {periph} transfers on {mcu}?",
    "How are interrupts handled for {periph} on {mcu}?",
    "What clock constraints apply to {periph} on {mcu}?",
    "How do I use {periph} in low-power Stop mode on {mcu}?",
    "What does STM32CubeMX generate for {periph} on {mcu}?",
    "How should I handle HAL errors from {periph} on {mcu}?",
    "How do I register custom callbacks for {periph} on {mcu}?",
]

CODE_SNIPPETS = {
    "UART": """
```c
UART_HandleTypeDef huart2;

void MX_USART2_UART_Init(void) {
  huart2.Instance        = USART2;
  huart2.Init.BaudRate   = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits   = UART_STOPBITS_1;
  huart2.Init.Parity     = UART_PARITY_NONE;
  huart2.Init.Mode       = UART_MODE_TX_RX;
  if (HAL_UART_Init(&huart2) != HAL_OK) { Error_Handler(); }
}
```""",
    "SPI": """
```c
SPI_HandleTypeDef hspi1;

void MX_SPI1_Init(void) {
  hspi1.Instance               = SPI1;
  hspi1.Init.Mode              = SPI_MODE_MASTER;
  hspi1.Init.Direction         = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize          = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity       = SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase          = SPI_PHASE_1EDGE;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16;
  if (HAL_SPI_Init(&hspi1) != HAL_OK) { Error_Handler(); }
}
```""",
    "I2C": """
```c
I2C_HandleTypeDef hi2c1;

void MX_I2C1_Init(void) {
  hi2c1.Instance              = I2C1;
  hi2c1.Init.ClockSpeed       = 100000;
  hi2c1.Init.DutyCycle        = I2C_DUTYCYCLE_2;
  hi2c1.Init.AddressingMode   = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode  = I2C_DUALADDRESS_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK) { Error_Handler(); }
}
```""",
}


def make_document(doc_id: int, periph: str, mcu: str) -> dict:
    topic_key = random.choice(list(TOPICS.keys()))
    topic_label = TOPICS[topic_key]

    # Build paragraphs (3-6 per document)
    n_para = random.randint(3, 6)
    selected = random.sample(PARAGRAPH_TEMPLATES, min(n_para, len(PARAGRAPH_TEMPLATES)))
    paragraphs = [t.format(periph=periph, mcu=mcu) for t in selected]

    # Add code snippet if available
    code = CODE_SNIPPETS.get(periph, "")

    # Build FAQ section
    faq_q = random.choice(FAQ_TEMPLATES).format(periph=periph, mcu=mcu)
    faq_a = paragraphs[0]  # reuse first paragraph as FAQ answer

    full_text = "\n\n".join(paragraphs)
    if code:
        full_text += f"\n\nExample code:\n{code}"

    return {
        "id": f"doc_{doc_id:04d}",
        "title": f"{mcu} {periph} — {topic_label}",
        "peripheral": periph,
        "mcu_family": mcu,
        "topic": topic_key,
        "content": full_text,
        "paragraphs": paragraphs,
        "code_snippet": code.strip() if code else None,
        "faq": {"question": faq_q, "answer": faq_a},
        "metadata": {
            "source": "STM32CubeXX_HAL_Driver",
            "version": f"v{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,9)}",
            "doc_type": "technical_note",
            "word_count": len(full_text.split()),
        },
    }


def generate_dataset(n_docs: int = 120, output_path: str = "data/stm32cube_kb.json") -> list:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    docs = []
    for i in range(n_docs):
        periph = random.choice(PERIPHERALS)
        mcu = random.choice(MCU_FAMILIES)
        docs.append(make_document(i, periph, mcu))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated {len(docs)} synthetic STM32Cube documents → {output_path}")
    return docs


# ── Ground-truth eval questions ────────────────────────────────────────────

EVAL_QUESTIONS = [
    # Peripheral init questions
    {"id": "q01", "question": "How do I initialize UART on STM32F4?",
     "relevant_topic": "init", "relevant_periph": "UART"},
    {"id": "q02", "question": "What is the SPI initialization sequence on STM32H7?",
     "relevant_topic": "init", "relevant_periph": "SPI"},
    {"id": "q03", "question": "How to configure I2C clock speed on STM32L4?",
     "relevant_topic": "clock", "relevant_periph": "I2C"},

    # DMA questions
    {"id": "q04", "question": "How do I set up DMA for UART transmit on STM32?",
     "relevant_topic": "dma", "relevant_periph": "UART"},
    {"id": "q05", "question": "Configure DMA for SPI peripheral on STM32H7",
     "relevant_topic": "dma", "relevant_periph": "SPI"},

    # Interrupt questions
    {"id": "q06", "question": "How to handle UART receive interrupt on STM32?",
     "relevant_topic": "irq", "relevant_periph": "UART"},
    {"id": "q07", "question": "Enable TIM interrupt with NVIC priority configuration",
     "relevant_topic": "irq", "relevant_periph": "TIM"},

    # Low-power questions
    {"id": "q08", "question": "Use UART in STM32 Stop mode for low power application",
     "relevant_topic": "power", "relevant_periph": "UART"},
    {"id": "q09", "question": "Low power I2C operation on STM32L4",
     "relevant_topic": "power", "relevant_periph": "I2C"},

    # CubeMX / code generation
    {"id": "q10", "question": "What code does STM32CubeMX generate for SPI initialization?",
     "relevant_topic": "init", "relevant_periph": "SPI"},
    {"id": "q11", "question": "MX_UART_Init function generated by STM32CubeMX",
     "relevant_topic": "init", "relevant_periph": "UART"},

    # Error handling
    {"id": "q12", "question": "How to handle HAL_TIMEOUT error from UART transmit?",
     "relevant_topic": "init", "relevant_periph": "UART"},
    {"id": "q13", "question": "HAL error codes and recovery strategy for SPI",
     "relevant_topic": "init", "relevant_periph": "SPI"},

    # Callbacks
    {"id": "q14", "question": "How to register a custom DMA transfer complete callback?",
     "relevant_topic": "dma", "relevant_periph": "DMA"},
    {"id": "q15", "question": "Override HAL_UART_TxCpltCallback for non-blocking transfer",
     "relevant_topic": "init", "relevant_periph": "UART"},

    # Clock
    {"id": "q16", "question": "Configure PLL for maximum frequency on STM32H7",
     "relevant_topic": "clock", "relevant_periph": "RCC"},
    {"id": "q17", "question": "ADC clock source configuration on STM32G0",
     "relevant_topic": "clock", "relevant_periph": "ADC"},

    # Security / advanced
    {"id": "q18", "question": "TrustZone configuration for secure boot on STM32U5",
     "relevant_topic": "security", "relevant_periph": "RCC"},
    {"id": "q19", "question": "FreeRTOS task scheduling with STM32 HAL",
     "relevant_topic": "rtos", "relevant_periph": "TIM"},
    {"id": "q20", "question": "Debug interface SWD configuration and trace on STM32F7",
     "relevant_topic": "debug", "relevant_periph": "GPIO"},
]


def save_eval_questions(output_path: str = "data/eval_questions.json"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(EVAL_QUESTIONS, f, indent=2)
    print(f"✅ Saved {len(EVAL_QUESTIONS)} eval questions → {output_path}")
    return EVAL_QUESTIONS


if __name__ == "__main__":
    generate_dataset(120)
    save_eval_questions()
