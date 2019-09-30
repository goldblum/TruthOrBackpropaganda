"""Training/Validation Routines."""


import torch


def train(net, optimizer, scheduler, loss_fn, trainloader, config, path=None, dryrun=False):
    """Standardized pytorch training routine."""
    net.train()
    net.to(**config['setup'])
    for epoch in range(config['epochs']):
        # Train
        epoch_loss = 0.0
        optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(trainloader):
            if not config['full_batch']:
                optimizer.zero_grad()
            inputs = inputs.to(device=config['setup']['device'], dtype=config['setup']['dtype'])
            targets = targets.to(device=config['setup']['device'], dtype=torch.long)
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            if not config['full_batch']:
                optimizer.step()
            epoch_loss += loss.item()

            if dryrun:
                break
        if config['full_batch']:
            optimizer.step()

        if epoch % config['print_loss'] == 0:
            print(f'Epoch loss in epoch {epoch} was {epoch_loss/(i+1):.12f}')

        scheduler.step()
        if epoch > config['switch_to_gd']:
            config['full_batch'] = True

        if epoch > config['stop_batchnorm']:
            net.eval()

        if (epoch % 50 == 0) and path is not None:
            torch.save(net.state_dict(), path + 'temp')

        if dryrun:
            break
    net.eval()


def distill(teacher, student, optimizer, scheduler, loss_fn, trainloader, config, path=None, dryrun=False):
    """Distill teacher onto student."""
    [(net.train(), net.to(**config['setup'])) for net in [teacher, student]]

    loss_cent = torch.nn.CrossEntropyLoss()

    offset = 100

    for epoch in range(config['epochs']):
        # Train
        epoch_loss = 0.0
        optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(trainloader):
            if not config['full_batch']:
                optimizer.zero_grad()
            inputs = inputs.to(device=config['setup']['device'], dtype=config['setup']['dtype'])
            outputs = student(inputs)
            loss = loss_fn(outputs.log_softmax(dim=1), teacher(inputs).softmax(dim=1))
            epoch_loss += loss.item()
            if config['gradpen'] > 0 and epoch > offset or dryrun:
                targets = targets.to(device=config['setup']['device'], dtype=torch.long)
                direct_loss = loss_cent(outputs, targets)
                grads = torch.autograd.grad(direct_loss, student.parameters(), only_inputs=True,
                                            create_graph=True, retain_graph=True)
                for grad in grads:
                    if config['gradual']:
                        damping = (epoch - gradient_penalty_offset) / (config['epochs'] - offset)
                        loss = loss + config['gradpen'] / 2 * grad.pow(2).sum() * damping
                    else:
                        loss = loss + config['gradpen'] / 2 * grad.pow(2).sum()
                # print(f'Mod loss is {loss.item()}')
            if config['centpen'] > 0 and epoch > offset or dryrun:
                targets = targets.to(device=config['setup']['device'], dtype=torch.long)
                direct_loss = loss_cent(outputs, targets)
                if config['gradual']:
                    damping = (epoch - offset) / (config['epochs'] - offset)
                    loss = (1 - damping) * loss + config['centpen'] * damping * direct_loss
                else:
                    loss = loss + config['centpen'] * direct_loss
            loss.backward()
            if not config['full_batch']:
                optimizer.step()

            if dryrun:
                break
        if config['full_batch']:
            optimizer.step()

        if epoch % config['print_loss'] == 0:
            print(f'Epoch loss in epoch {epoch} was {epoch_loss/(i+1):.12f}')

        scheduler.step()

        if dryrun:
            break
    [net.eval() for net in [teacher, student]]


def get_accuracy(net, dataloader, config):
    """Get accuracy of net relative to dataloader."""
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device=config['setup']['device'], dtype=config['setup']['dtype'])
            targets = targets.to(device=config['setup']['device'], dtype=torch.long)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total


def compute_loss(net, loss_fn, dataloader, config, add_weight_decay=True):
    """Compute loss, with or without weight decay taken into account."""
    net.eval()
    with torch.no_grad():
        loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device=config['setup']['device'], dtype=config['setup']['dtype'])
            targets = targets.to(device=config['setup']['device'], dtype=torch.long)
            outputs = net(inputs)
            loss += loss_fn(outputs, targets).item()
            if add_weight_decay:
                if config['weight_decay'] > 0:
                    for param in net.parameters():
                        loss += config['weight_decay'] * 0.5 * param.pow(2).sum().item()
    return loss / (i + 1)
