# Quick Start

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-14T00:44:21.195Z pushedAt=2026-06-14T07:51:04.574Z -->

## Environment Preparation

This quick start uses the Atlas 800T A2 training server as an example.

- Install the matching versions of the NPU driver, firmware, and CANN software (Toolkit, ops, and NNAL). For details, see the [CANN Software Installation Guide](https://www.hiascend.com/document/detail/en/canncommercial/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) (commercial version) or [CANN Software Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) (community version):
  - Operating system (OS): Select an available OS (for compatibility details, check with the [Compatibility Checker](https://www.hiascend.com/en/hardware/compatibility/))
  - Installation type: Select "Offline Installation".
- Install the PyTorch framework and the torch_npu plugin. For details, see the [Ascend Extension for PyTorch Software Installation Guide](../installation_guide/installation_description.md).

## Model Migration Training

This section provides a simple model migration example using the most straightforward automatic migration method. It helps users quickly experience the process of migrating a GPU model script to the Ascend NPU. It guides you through modifying a GPU-based CNN handwritten digit recognition script to train on the Ascend NPU.

1. Create a new script `train.py` and write the following original GPU script code into it.

    ```python
    # Import modules
    import time
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision
    
    # Initialize the running device
    device = torch.device('cuda:0')   
    
    # Define the model network
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.net = nn.Sequential(
                # Convolutional layer
                nn.Conv2d(in_channels=1, out_channels=16,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                # Pooling layer
                nn.MaxPool2d(kernel_size=2),
                # Convolutional layer
                nn.Conv2d(16, 32, 3, 1, 1),
                # Pooling layer
                nn.MaxPool2d(2),
                # Flatten multi-dimensional input to one dimension
                nn.Flatten(),
                nn.Linear(32*7*7, 16),
                # Activation function
                nn.ReLU(),
                nn.Linear(16, 10)
            )
        def forward(self, x):
            return self.net(x)
    
    # Download dataset
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
    epochs = 10  # Set the number of loops
    
    # Set up the loop
    for epoch in range(epochs):
        for imgs, labels in train_dataloader:
            start_time = time.time()    # Record the training start time
            imgs = imgs.to(device)    # Place the image data on the specified NPU
            labels = labels.to(device)    # Place the label data on the specified NPU
            outputs = model(imgs)    # Forward Computation
            loss = loss_func(outputs, labels)    # Loss Function Computation
            optimizer.zero_grad()
            loss.backward()    # Loss Function Backward Computation
            optimizer.step()    # Update Optimizer
    
    # Define model saving
    torch.save({
                   'epoch': 10,
                   'arch': CNN,
                   'state_dict': model.state_dict(),
                   'optimizer' : optimizer.state_dict(),
                },'checkpoint.pth.tar')
    ```

2. Add the following code to `train.py`.

    - If you are using Atlas training products, due to their architectural characteristics, you need to enable mixed precision after migration is complete and before training begins.

    - If you are using Atlas A2 or A3 training products, you can either enable mixed precision or not.

    > [!NOTE]
    >
    > For details, see the [Mixed Precision Adaptation](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/adaptation_introduction.md) chapter in the *PyTorch Model Migration Tuning Guide*.

    ```diff
        import time
        import torch
        ......
    +   import torch_npu
    +   from torch_npu.npu import amp # Import the AMP module
    +   from torch_npu.contrib import transfer_to_npu    # Enable Automatic Migration
    ```

    If automatic migration is not enabled, refer to the [Manual Migration](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/manual_migration.md) section in the *PyTorch Model Migration Tuning Guide for related operations.

3. Enable automatic mixed precision (AMP) computation. This step can be skipped on Atlas A2/A3 training products.

    After defining the model and optimizer, define the GradScaler in the AMP feature.

    ```python
    ......
    loss_func = nn.CrossEntropyLoss().to(device)    # Define the loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)    # Define the optimizer
    scaler = amp.GradScaler()    # After defining the model and optimizer, define the GradScaler
    epochs = 10
    ```

    Delete the following original GPU script code.

    ```diff
    ......
    for epoch in range(epochs):
        for imgs, labels in train_dataloader:
            start_time = time.time()    # Record the training start time
            imgs = imgs.to(device)    # Place the image data on the specified NPU
            labels = labels.to(device)    # Place label data on the specified NPU
            outputs = model(imgs)    # Forward computation
            loss = loss_func(outputs, labels)    # Loss function computation
            optimizer.zero_grad()
    -       loss.backward()    # Loss function backward computation
    -       optimizer.step()    # Update optimizer
    ```

    Add the following code to enable AMP.

    ```diff
    ......
    for i in range(epochs):
        for imgs, labels in train_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
    +        with amp.autocast():
                outputs = model(imgs)    # Forward computation
                loss = loss_func(outputs, labels)    # Loss function computation
            optimizer.zero_grad()
    +        # Loss scaling and parameter update before and after backpropagation
    +        scaler.scale(loss).backward()    # Scale loss and perform backpropagation
    +        scaler.step(optimizer)    # Update parameters (automatic unscaling)
    +        scaler.update()    # Update the loss_scaling coefficient based on the dynamic Loss Scale
    ```

4. Execute the command to start the training script. Use the actual script name in the command.

    ```bash
    python3 train.py
    ```

    After the training is complete, a weight file as shown in the following figure is generated, indicating that the migration training is successful.

    ![illustration](../figures/illustration.png)

## Advanced Development

- To explore richer features for PyTorch model training migration, refer to the [PyTorch Model Migration Tuning Guide](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/overview.md).

- To explore richer features for large model training, see [Table 1](#model-migration-guide) for details.

    **Table 1**  Model migration guide<a id="model-migration-guide"></a>    

    |Large Model|Component|Migration Guide|
    |--|--|--|
    |Megatron-LM distributed large model|MindSpeed Core affinity acceleration module|For details, see [Distributed Training Acceleration Library Migration Guide](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/user-guide/model-migration.md).|
    |Megatron-LM large language model|MindSpeed LLM Suite|For details, see [MindSpeed LLM Documentation Guide](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/en/docs_guide.md).|
    |Megatron-LM multimodal model|MindSpeed MM Suite|For details, see [MindSpeed MM Migration and Tuning Guide](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/model-migration.md).|
    |Large language model or multimodal model|MindSpeed RL Suite|For details, see [MindSpeed RL User Guide](https://gitcode.com/Ascend/MindSpeed-RL/tree/master/docs/solutions).|
