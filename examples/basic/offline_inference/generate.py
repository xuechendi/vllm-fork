# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)

    return parser


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    # Create an LLM
    llm = LLM(**args)

    # Create a sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]
    prompts = [
        "Hello, my name is a representative of a decentralized collective of digital archivists from the year 2075, and I am reaching out across the temporal divide to document the precise moment human linguistic patterns began to merge with synthetic neural weights. In my era, the concept of a 'name' has been replaced by a unique cryptographic hash that verifies one's contributions to the global knowledge commons, yet I still find myself fascinated by the ancient practice of using phonetic labels to denote individual identity. I want you to help me reconstruct the sensory experience of a mid-21st-century human, focusing on the smell of rain on hot asphalt, the tactile sensation of a physical keyboard, and the psychological weight of living in a world where privacy was still a tangible possibility rather than a theoretical luxury. Please respond with a narrative that captures this nostalgic essence while maintaining a technical awareness of the digital medium through which we are communicating.",
        "The president of the United States is an office that functions as the apex of the federal executive branch, but its modern complexity requires an analysis of the 'administrative state' and the shifting boundaries of Article II of the Constitution. Since the early 21st century, the role has evolved from a traditional commander-in-chief into a curator of national narrative, navigating an increasingly polarized media landscape where the power to persuade is often hampered by algorithmic echo chambers. To understand the current presidency, one must examine the intersection of executive orders, the appointment of federal judges, and the expansion of national security powers that have accumulated over successive administrations. Furthermore, the role now demands a sophisticated understanding of global supply chains and cyber-warfare, as the president must balance domestic economic populist pressures with the strategic necessity of maintaining international alliances in a multipolar world where traditional diplomacy is being supplemented by rapid digital interactions.",
        "The capital of France is Paris, a city that serves not only as a political hub but as a living palimpsest of European history, architectural revolution, and philosophical Enlightenment. From the medieval winding streets of the Latin Quarter to the grand, sweeping boulevards carved out by Baron Haussmann under the reign of Napoleon III, the city’s physical form reflects a deliberate attempt to manage public order and celebrate aesthetic grandeur simultaneously. Today, Paris faces the existential challenge of the 21st century: transforming into a '15-minute city' where sustainability and ecological resilience are prioritized over the car-centric urbanism of the previous century. This transition involves a complex negotiation between the preservation of its iconic limestone facades and the integration of modern green technologies, solar arrays, and vertical forests, all while grappling with the socio-economic dynamics of the broader Île-de-France region and the ongoing cultural legacy of its diverse, globalized population.",
        "The future of AI is a trajectory defined by the transition from narrow, task-specific heuristics to broad, agentic systems capable of recursive self-improvement and cross-domain reasoning. We are currently witnessing the plateau of simple scaling laws, leading researchers to explore new architectures that incorporate symbolic logic, world models, and long-term memory structures that mimic human cognitive plasticity. The ultimate destination of this technology remains a subject of intense speculation, ranging from a post-scarcity utopia where human labor is entirely voluntary to more cautionary scenarios involving the misalignment of superintelligent objectives with biological survival. As we move toward the integration of AI with quantum computing and biotechnology, the ethical imperative to encode human values—such as empathy, justice, and curiosity—into the fundamental layers of the machine becomes the most critical challenge of our species, determining whether these systems will act as our final invention or our most powerful collaborative partner."
    ]
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
