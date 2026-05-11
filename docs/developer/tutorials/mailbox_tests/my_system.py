import cpu.iss.riscv
import memory.memory
import vp.clock_domain
import interco.router
import utils.loader.loader
import gvsoc.systree
import gvsoc.runner
import devices.mailbox.mailbox
import cpu.plic


GAPY_TARGET = True

class Soc(gvsoc.systree.Component):

    def __init__(self, parent, name, parser):
        super().__init__(parent, name)

        [args, __] = parser.parse_known_args()

        binary = args.binary

        ico = interco.router.Router(self, 'ico')

        mem = memory.memory.Memory(self, 'mem', size=0x00100000)
        ico.o_MAP(mem.i_INPUT(), 'mem', base=0x00000000, size=0x00100000, rm_base=True)

        #PLIC:->
        plic = cpu.plic.Plic(self, 'plic', ndev=32)
        ico.o_MAP(plic.i_INPUT(), 'plic', base=0x0C000000, size=0x04000000, rm_base=True)

        

        # Mailbox
        comp = devices.mailbox.mailbox.Mailbox(self, 'mailbox', size=10)
        ico.o_MAP(comp.i_INPUT(), 'comp', base=0x20000000, size=0x00001000, rm_base=True)

        #Mailbox to PLIC
        comp.o_RCV_IRQ(1, plic.i_IRQ(0))


        #Core 0 -> sender
        core0 = cpu.iss.riscv.Riscv(self, 'core0', isa='rv64imafdc', core_id=0)
        core0.o_FETCH(ico.i_INPUT())
        core0.o_DATA(ico.i_INPUT())
        
        plic.o_M_IRQ(core=0,itf=core0.i_IRQ(11))

        # Core 1 -> receiver
        core1 = cpu.iss.riscv.Riscv(self, 'core1', isa='rv64imafdc', core_id=1)
        core1.o_FETCH(ico.i_INPUT())
        core1.o_DATA(ico.i_INPUT())
        plic.o_M_IRQ(core = 1,itf=core1.i_IRQ(11))

        
        loader = utils.loader.loader.ElfLoader(self, 'loader', binary=binary)
        loader.o_OUT(ico.i_INPUT())
        loader.o_START(core0.i_FETCHEN()) # Start Core 0
        loader.o_START(core1.i_FETCHEN()) # Start Core 1
        loader.o_ENTRY(core0.i_ENTRY())
        loader.o_ENTRY(core1.i_ENTRY())

class Rv64(gvsoc.systree.Component):

    def __init__(self, parent, name, parser, options):

        super().__init__(parent, name, options=options)

        clock = vp.clock_domain.Clock_domain(self, 'clock', frequency=100000000)
        soc = Soc(self, 'soc', parser)
        clock.o_CLOCK    ( soc.i_CLOCK     ())


class Target(gvsoc.runner.Target):

    def __init__(self, parser, options):
        super(Target, self).__init__(parser, options,
            model=Rv64, description="RV64 virtual board")

