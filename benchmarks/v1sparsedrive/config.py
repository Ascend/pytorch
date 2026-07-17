CONFIG = dict(
    seed=3407,
    batch_size=2,
    num_workers=0,
    max_epochs=1,
    train_steps=200,
    lr=1e-4,
    weight_decay=1e-4,

    model=dict(
        num_det_classes=10,
        num_map_classes=3,
        backbone_pretrained=None,
        use_grid_mask=True,
    ),

    dummy_dataset=dict(
        length=400,
        image_size=(256, 704),
        num_det_boxes=20,
        num_map_instances=10,
        num_det_classes=10,
        num_map_classes=3,
    )
)
