"""Minimal standalone cstorch training loop — DIAGNOSTIC ONLY.

This is the canonical Cerebras custom-loop example (a 2-layer MLP) with synthetic
data, run as a *standalone script* (not cszoo fit / the modelzoo Trainer). It
exists to answer one question:

    Does a raw standalone cstorch loop compile+execute on this CS-3 cluster,
    or does only the modelzoo Trainer path work?

Our real model (train_cstorch.py) fails with "Cannot compile empty CIRH module".
The modelzoo gpt3 example (cszoo fit) compiles fine on the same venv+cluster.
The difference is Trainer vs raw loop. If THIS minimal raw loop also fails empty,
the problem is the standalone-loop mechanism (nothing model-specific). If it
compiles+runs, the problem is specific to our model and we bisect that.

Nothing here imports from vqvae/ — it depends only on cstorch + torch.

Run via:  bash cerebras/submit_alcf.sh minprobe
"""
import os

import torch
import torch.nn.functional as F

import cerebras.pytorch as cstorch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(784, 256)
            self.fc2 = torch.nn.Linear(256, 10)

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)          # raw logits (CrossEntropyLoss expects them)

    # Same cluster-config shape train_cstorch uses (mgmt/creds come from the
    # ALCF user node's /opt/cerebras/config_v2).
    cluster_config = cstorch.distributed.ClusterConfig(
        num_csx=1, job_labels=["name=minprobe"], job_time_sec=3600,
        mount_dirs=[REPO_ROOT], python_paths=[REPO_ROOT],
    )
    backend = cstorch.backend("CSX", cluster_config=cluster_config)

    model = MLP()
    compiled_model = cstorch.compile(model, backend)
    optimizer = cstorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def input_fn():
        # A real torch DataLoader (as the canonical example uses), over a
        # synthetic in-memory dataset — no downloads, no data/ files.
        n = 640
        x = torch.randn(n, 1, 28, 28)
        y = torch.randint(0, 10, (n,), dtype=torch.int32)
        ds = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    dataloader = cstorch.utils.data.DataLoader(input_fn)
    loss_fn = torch.nn.CrossEntropyLoss()

    @cstorch.trace
    def training_step(inputs, targets):
        outputs = compiled_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    @cstorch.step_closure
    def print_loss(loss, step):
        print(f"step {step}: loss={loss.item():.4f}")

    executor = cstorch.utils.data.DataExecutor(dataloader, num_steps=5)
    model.train()
    step = 0
    print("minimal_probe: starting standalone cstorch loop...")
    for inputs, targets in executor:
        loss = training_step(inputs, targets)
        print_loss(loss, step)
        step += 1
    print("minimal_probe: DONE — standalone loop compiled and executed.")


if __name__ == "__main__":
    main()
