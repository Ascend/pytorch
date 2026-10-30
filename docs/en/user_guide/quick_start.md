# Quick Start

This document uses a CNN model for recognizing MNIST handwritten digits as an example to demonstrate the process of migrating a GPU training script to the Ascend NPU through automatic migration.

## Environment Preparation

This example uses the Atlas 800T A2 training server as an example.

Complete the following installation and environment configuration first. If you have already done so, you can proceed directly to data preparation.

- Install the matching versions of the NPU driver firmware and CANN software (Toolkit, ops, and NNAL). For details, see [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum), and load the environment variables as instructed in the guide.
- PyTorch framework and TorchNPU plugin: For details, see [Quick Installation](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=180&ids=89dda9ba9de741349efa03687a487678%2C204%2C200%2C1%2C6%2C177%2C).
- Install the torchvision that matches the PyTorch version. See [Installing torchvision](https://pytorch.org/get-started/locally/).

## Data Preparation

This example uses the MNIST handwritten digit dataset. Prepare the test dataset based on your network environment:

- **Method 1**: If the training server has normal network access, on the first run, the [script](#script) downloads data through `download=True`. The data is saved in `mnist/MNIST/raw/` under the current working directory. After the data is fully downloaded, subsequent runs reuse the local data.

- **Method 2**: If the training server has no network access, you can first run the following command in a networked environment with torchvision installed, then copy the entire generated `mnist` directory to the working directory where the [script](#script) runs on the training server, and change `download=True` to `download=False` in the training script.

    ```bash
    python3 -c "from torchvision.datasets import MNIST; MNIST(root='mnist', train=True, download=True)"
    ```

## Model Migration Training

> [!NOTE]
>
> - Complete the applicable modifications in Steps 1 to 3 first, and then perform Step 4. Choose whether to enable mixed precision according to the hardware applicability descriptions in Step 3.
> - In the `diff` code blocks below, `+` indicates added lines, `-` indicates deleted lines, and `...` indicates omitted original code. These markers are used to indicate where changes were made and should not be written into `train.py`. Ordinary `python` code blocks can be copied directly; keep the code indentation.

1. Create a new script train.py and write the following original GPU script code into it. <a id="script"></a>

    ```python
    # Import modules
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torchvision
    
    # Initialize the running device
    device = torch.device('cuda:0')   
    
    # Define the model network
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.net = nn.Sequential(
                # Convolution layer
                nn.Conv2d(in_channels=1, out_channels=16,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                # Pooling layer
                nn.MaxPool2d(kernel_size=2),
                # Convolution layer
                nn.Conv2d(16, 32, 3, 1, 1),
                # Pooling layer
                nn.MaxPool2d(2),
                # Flatten the multi-dimensional input
                nn.Flatten(),
                nn.Linear(32*7*7, 16),
                # Activation function
                nn.ReLU(),
                nn.Linear(16, 10)
            )
        def forward(self, x):
            return self.net(x)
    
    # Download the dataset
    train_data = torchvision.datasets.MNIST(
        root='mnist',
        download=True,
        train=True,
        transform=torchvision.transforms.ToTensor()
    )
    
    # Define training-related parameters
    batch_size = 64   
    model = CNN().to(device)  # Define the model
    train_dataloader = DataLoader(train_data, batch_size=batch_size)    # Define the DataLoader
    loss_func = nn.CrossEntropyLoss().to(device)    # Define the loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)    # Define the optimizer
    epochs = 10  # Set the number of epochs
    
    # Set up the loop
    for epoch in range(epochs):
        for step, (imgs, labels) in enumerate(train_dataloader, start=1):
            imgs = imgs.to(device)    # Place the image data on the specified GPU
            labels = labels.to(device)    # Place the label data on the specified GPU
            outputs = model(imgs)    # Forward computation
            loss = loss_func(outputs, labels)    # Compute the loss
            optimizer.zero_grad()
            loss.backward()    # Backward computation of the loss function
            optimizer.step()    # Update the optimizer
            if step % 100 == 0 or step == len(train_dataloader):
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{step}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
    
    # Save the model checkpoint
    torch.save({
                   'epoch': epochs,
                   'state_dict': model.state_dict(),
                   'optimizer' : optimizer.state_dict(),
                },'checkpoint.pth.tar')
    print("Training completed. Checkpoint saved to checkpoint.pth.tar")
    ```

2. Add the following code to the import area at the top of `train.py`, ensuring that it comes before `device = torch.device('cuda:0')`.

    ```diff
     import torch
     import torch.nn as nn
     from torch.utils.data import DataLoader
     import torchvision
    +import torch_npu
    +from torch_npu.contrib import transfer_to_npu    # Enable automatic migration
    ```

    Importing `transfer_to_npu` enables automatic migration, mapping the relevant CUDA calls in this example to the NPU. Therefore, `cuda:0` in the script can be retained, and the model and data actually run on the NPU. If automatic migration is not enabled, refer to the [Manual Migration](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/docs/en/model_migration/manual_migration.md) document to complete the migration.

3. Add the following code to train.py to enable AMP (Automatic Mixed Precision).

    > [!NOTE]
    >
    > <!-- npu="910" id1 -->
    > - If you use <term>Atlas training products</term>, due to their architectural characteristics, you must perform this step to enable mixed precision after the migration is complete and before training begins.
    > <!-- end id1 -->
    > <!-- npu="950,A3,910b" id2 -->
    > - If you use <term>Atlas A2 training products</term>, <term>Atlas A3 training products</term>, or <term>Ascend 950DT</term>, you can choose whether to enable mixed precision. If you choose not to enable it, you can skip this step.
    > <!-- end id2 -->
    > - For details about mixed precision, see [Mixed Precision Adaptation](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/docs/en/mixed_precision_adaptation/adaptation_introduction.md).

    AMP involves two main operations: `autocast` selects the appropriate computation precision for operators in forward computation, and `GradScaler` reduces the risk of underflow in low-precision gradients by scaling the loss, and restores the gradient scale during parameter updates.

    1. Add the AMP module in the import area, and define `GradScaler` after the model and optimizer definitions and before the training loop:

        ```diff
        import torch_npu
        +from torch_npu.npu import amp
        from torch_npu.contrib import transfer_to_npu
        ...
        loss_func = nn.CrossEntropyLoss().to(device)    # Define the loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)    # Define the optimizer
        +scaler = amp.GradScaler()    # Define the GradScaler
        epochs = 10
        ```

    2. Replace the entire training loop starting from `for epoch in range(epochs):` in Step 1 with the following code, and keep the model saving code after the loop in its original position. Note that forward computation and loss computation are both placed inside `with amp.autocast():`, and backward propagation is executed outside it.

        ```python
        for epoch in range(epochs):
            for step, (imgs, labels) in enumerate(train_dataloader, start=1):
                imgs = imgs.to(device)  # Place the image data on the specified NPU
                labels = labels.to(device)  # Place the label data on the specified NPU
                with amp.autocast():
                    outputs = model(imgs)    # Forward computation
                    loss = loss_func(outputs, labels)    # Compute the loss
                optimizer.zero_grad()
                scaler.scale(loss).backward()    # Scale the loss and backpropagate
                scaler.step(optimizer)    # Restore the gradient scale and update parameters; skip the update if gradients are not finite
                scaler.update()    # Dynamically update the scaling factor
                if step % 100 == 0 or step == len(train_dataloader):
                    print(f"Epoch [{epoch + 1}/{epochs}], Step [{step}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
        ```

4. Run the following command in the directory where `train.py` is located to start training. For the first experience, you can change `epochs = 10` to `epochs = 1` to first verify one round of the training flow.

    ```bash
    python3 train.py
    ```

    During training, the current epoch, training step, and loss are printed every 100 batches and at the last batch of each epoch, making it easy to observe the training progress. Here, loss is the loss value of the current batch.

    With default parameters, the log format is as follows (the loss values are for illustration only; actual results vary with each run):

    ```text
    Epoch [1/10], Step [100/938], Loss: 0.4123
    Epoch [1/10], Step [200/938], Loss: 0.2856
    ...
    Epoch [10/10], Step [938/938], Loss: 0.0732
    Training completed. Checkpoint saved to checkpoint.pth.tar
    ```

    The loss may fluctuate across batches and is not required to decrease every time. If `nan` or `inf` appears, check the data, learning rate, and mixed precision configuration.

    If the training completes normally, a saving-complete message is printed, and `checkpoint.pth.tar` is generated or updated in the current working directory, it indicates that the migration training flow in this example is working. The file contains information such as model parameters, optimizer state, and the number of training epochs. Judge based on the logs of this run and the file modification time to avoid mistaking an old file for the current result. Model recognition performance still needs to be evaluated on an independent test set; this example does not include an evaluation step.

    ![Checkpoint file generated by training](../figures/illustration.png)

## Advanced Development

- If you want to explore more features of PyTorch model training migration, go to the [PyTorch Training Model Migration and Tuning Guide](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/docs/en/_menu.md) document for details.
- If you want to explore more features of large model training, see the following table for more information:

    | Large Model | Component | Migration Guide |
    |--|--|--|
    | Megatron-LM distributed large model | MindSpeed Core affinity acceleration module | See the [Distributed Training Acceleration Library Migration Guide](https://gitcode.com/Ascend/MindSpeed/blob/26.2.0_core_r0.12.1/docs/en/user-guide/model-migration.md). |
    | Megatron-LM large language model | MindSpeed LLM Suite | See the [MindSpeed LLM Documentation Guide](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.2.0/docs/en/docs_guide.md). |
    | Megatron-LM multimodal model | MindSpeed MM Suite | See the [MindSpeed MM Migration and Tuning Guide](https://gitcode.com/Ascend/MindSpeed-MM/blob/26.2.0/docs/en/guides/development/model-migration.md). |
    | Large language model or multimodal model | veRL suite | See the [veRL Migration Guide](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/en/dev_guide/model_dev/transfer_to_npu_guide.md). |
