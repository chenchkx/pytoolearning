

import pandapower
import pandapower.networks as pn

net = pn.case39()

pandapower.runopp(net)

