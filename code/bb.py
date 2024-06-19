## 79 ########################################################################

print(__file__)
assert 'project_' in __file__
from utilz2 import *
import sys,os
from projutils import *
from ..params.a import *
from .dataloader import *

thispath=pname(pname(__file__))
sys.path.insert(0,opj(thispath,'env'))
figures_path=opj(thispath,'figures')
mkdirp(figures_path)

device = torch.device(device if torch.cuda.is_available() else 'cpu')

net=get_net(
    device=device,
    run_path=run_path,
)

#fig.suptitle('tac_attn - bb.py')
dataiter = iter(trainloader)
for i in range(100):
    ms=[]
    oimages, labels = next(dataiter)
    print(oimages.size(),oimages.max(),oimages.min())
    oimages=oimages.to(device)
    print(labels)
    #sh(cuda_to_rgb_image(oimages[0,:]),2)
    imgdic={}
    outdic={}
    vals=[10,12,14,16]
    for w in vals:
        for h in vals:
            m=np.zeros((32,32))
            for x in range(32):
                for y in range(32):
                    if x-w<0:
                        continue
                    if y-h<0:
                        continue                    
                    if x+w>32:
                        continue
                    if y+h>32:
                        continue
                    images=1*oimages
                    images=F.interpolate(
                        images[:,:,x-w:x+w,y-h:y+h],
                        size=(32,32),
                        mode='bilinear',
                        align_corners=False)
                    outputs=net(images).detach().cpu().numpy()
                    outdic[x,y,w,h]=outputs[0,labels.item()][0][0]
                    imgdic[x,y,w,h]=1*images
    bestout=-1
    ov=[]
    for k in outdic:
        ov.append(outdic[k])
        #print(outdic[k],bestout,outdic[k]>bestout)
        if outdic[k]>bestout:
            bestout=outdic[k]
            bestxy=k
    #print(outdic)
    #sh(imgdic[bestxy],4)
    CA()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    a=cuda_to_rgb_image(imgdic[bestxy])
    #sh(a,4)
    ax2.imshow(a)
    x,y,w,h=bestxy
    #sh(cuda_to_rgb_image(oimages[0,:]),10)
    b=cuda_to_rgb_image(oimages[0,:])
    #sh(b,10)
    ax1.imshow(b)
    #figure(10)
    ax1.plot(na([y-h,y-h,y+h,y+h,y-h])-0.5,na([x-w,x+w,x+w,x-w,x-w])-0.5,'r')
    savefigs(figures_path)
    spause()
    cm()
    
print('*** Done')

#EOF