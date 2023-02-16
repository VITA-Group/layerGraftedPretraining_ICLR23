import torch
from .utils import AverageMeter

@torch.no_grad()
def speed_test(train_loader, model, args, log):
    # speed test happens on single gpu
    assert torch.distributed.get_world_size() == 1

    # pre-warm card by runing with random input
    iter_loader = iter(train_loader)
    for i in range(100):
        try:
            images, target = next(iter_loader)
        except:
            iter_loader = iter(train_loader)
            images, target = next(iter_loader)
        images = images.cuda(args.local_rank, non_blocking=True)
        target = target.cuda(args.local_rank, non_blocking=True)
        model(images)

    timeMeter = AverageMeter()

    model.eval()
    for i, (images, target) in enumerate(train_loader):
        num_frames = images.shape[0]

        # measure data loading time
        images = images.cuda()
        target = target.cuda()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # compute output
        start.record()
        output = model(images)
        end.record()
        torch.cuda.synchronize()
        seq_total_time = start.elapsed_time(end)

        timeMeter.update(seq_total_time / num_frames, n=num_frames)

        if i % 50 == 0:
            print("cnt is {}, elapse time is {}({})".format(i, seq_total_time / num_frames, timeMeter.avg))
            log.info("inference time per frame avg is {:.2f} FPS".format(1000 / timeMeter.avg))
