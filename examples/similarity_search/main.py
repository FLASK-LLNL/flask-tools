import argparse
import asyncio
import os
from charge.tasks.task import Task
from charge.clients.client import Client
from charge.clients.autogen import AutoGenBackend


class RetrosynthesisTask(Task):
    def __init__(self, lead_molecules: list[str], **kwargs):
        mols = "\n".join(lead_molecules)
        system_prompt = (
            "You are an expert chemist. Your task is to perform retrosynthesis for a target molecule. "
            "If any tool is used, show the original tool output, followed by your overall answer. "
            "Provide your answer in a clear and concise manner. "
        )
        user_prompt = (
            "Use RAG and other available tools to find synthesis routes to make the following product molecule:\n"
            # "Find examples of reactions for similar molecules:\n"
            f"{mols}"
        )
        super().__init__(system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)
        self.lead_molecules = lead_molecules
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt


class ForwardSynthesisTask(Task):
    def __init__(self, lead_molecules: list[str], **kwargs):
        mols = "\n".join(lead_molecules)
        system_prompt = (
            "You are an expert chemist. Your task is to perform forward synthesis for a set of reactant molecules. "
            "If any tool is used, show the original tool output, followed by your overall answer. "
            "Provide your answer in a clear and concise manner. "
        )
        user_prompt = (
            "Use RAG and other available tools to predict forward synthesis from the following reactant molecules:\n"
            f"{mols}"
        )
        super().__init__(system_prompt=system_prompt, user_prompt=user_prompt, **kwargs)
        self.lead_molecules = lead_molecules
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        print("=" * 100)
        print("user prompt is", user_prompt)


def main():
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead-molecules", nargs="+", default=["CC(=O)O[C@H](C)CCN"])
    parser.add_argument(
        "--retrosynthesis",
        action="store_true",
        default=False,
        help="Whether to perform a retrosynthesis task.",
    )

    Client.add_std_parser_arguments(parser)
    args = parser.parse_args()

    agent_backend = AutoGenBackend(model=args.model, backend=args.backend)

    task = (
        RetrosynthesisTask(args.lead_molecules, server_urls=args.server_urls)
        if args.retrosynthesis
        else ForwardSynthesisTask(args.lead_molecules, server_urls=args.server_urls)
    )
    runner = agent_backend.create_agent(task=task)

    results = asyncio.run(runner.run())
    print(f"[{args.model} orchestrated] Task completed. Results: {results}")


if __name__ == "__main__":
    main()
