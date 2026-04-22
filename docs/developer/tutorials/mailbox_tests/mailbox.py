import gvsoc.systree

class Mailbox(gvsoc.systree.Component):
    """

    """
    def __init__(self, parent: gvsoc.systree.Component, name: str, size: int):

        super().__init__(parent, name)

        self.add_sources(['mailbox.cpp'])

        self.add_properties({
            "size": size
        })

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature='io')

    # The IRQ wire output
    def o_SND_IRQ(self,id:int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'irq_snd_{id}', itf, signature='wire<bool>')
    
    def o_RCV_IRQ(self,id:int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'irq_rcv_{id}', itf, signature='wire<bool>')