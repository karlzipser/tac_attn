
print(__file__)
from utilz2 import *
################################################################################
##
weights_file=''
figure_file=''
stats_file=''
if 'project_' in __file__:
    import sys,os
    sys.path.insert(0,os.path.join(pname(pname(__file__)),'env'))
    weights_file=opj(pname(pname(__file__)),'net/weights',d2p(time_str(),'pth'))
    figure_file=opj(pname(pname(__file__)),'figures',d2p(time_str(),'pdf'))
    stats_file=opj(pname(pname(__file__)),'stats',d2p(time_str(),'txt'))
##
################################################################################
from projutils import *
from ..params.a import *
from .dataloader import *
from .stats import *
from ..net.code.net import *

device = torch.device(device if torch.cuda.is_available() else 'cpu')

net=get_net(
    device=device,
    run_path='project_tac/15Jun24_10h16m58s-jake0',
)


dataiter = iter(testloader)
for i in range(100):
    images, labels = next(dataiter)
    sh(torchvision.utils.make_grid(images),1)
    plt.savefig(figure_file)
    images=images.to(device)
    m=get_blank_rgb(32,32)
    for x in range(32):
        for y in range(32):
            images=1*images
            images[:,:,x,y]=2
            outputs=net(images).detach().cpu().numpy()
            #print(outputs.size())
            m[x,y]=outputs[0,0]#.sum()
    spause()
    sh(z55(m),2,r=1)


print('*** Done')

#EOF