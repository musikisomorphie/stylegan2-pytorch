import synapseclient
import synapseutils

syn = synapseclient.Synapse()
syn.login('IID_learning', 'Newton314159')
files = synapseutils.syncFromSynapse(syn, 'syn17865732')
