import time
import torch

bs = [1, 4, 8, 16, 32, 64, 128, 256]
times = []
for i in bs:
    print(i)
    t = torch.rand(i, 1024, 768, device='cuda')
    torch.save(i, 'somefile.pt')

    start = time.time()
    for i in range(100):
        j = torch.load('somefile.pt', map_location='cuda')

    end = time.time()
    times.append(end - start)

print(bs)
print(times)


