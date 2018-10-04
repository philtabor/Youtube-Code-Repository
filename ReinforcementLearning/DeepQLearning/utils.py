import matplotlib.pyplot as plt 

def plotLearning(x, scores, epsilons, filename):   
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    ax2.scatter(x, scores, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)    
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1") 
    ax2.set_ylabel('Score', color="C1")       
    #ax2.xaxis.set_label_position('top') 
    ax2.yaxis.set_label_position('right') 
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)