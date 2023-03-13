import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 1e-4
batch_size = 32
num_epochs = 50

model = CarDetector(num_classes=3).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)

criterion = nn.BCEWithLogitsLoss()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

log_path = './logs/'
ckpt_path = './checkpoints/'