# %%
from bdahlen import makePlots, Potential
import warnings
warnings.filterwarnings("ignore")
# %%
if __name__ == '__main__':
    V = Potential()
    makePlots(V, [1,2,3,4])
# %%
