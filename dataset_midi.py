#%%
"""
###################

Midi data pre-processing 

###################
"""

import os
import glob
import librosa
import pretty_midi as pm
import matplotlib.pyplot as plt
import numpy as np

#%% Handle programs
nbInsHandled = 8

piano = np.linspace(0, 7, 8)
percussion = np.append(np.linspace(8, 15, 8), np.linspace(112, 119, 8))
organ = np.linspace(16, 23, 8)
guitar = np.linspace(24, 31, 8)
bass = np.linspace(32, 39, 8)
string = np.linspace(40, 47, 8)
ensemble = np.linspace(48, 55, 8)
brass = np.linspace(56, 63, 8)
reed = np.linspace(64, 71, 8)
pipe = np.linspace(72, 79, 8)
synth = np.linspace(80, 103, 24)



#progList = [piano, percussion, organ, guitar, bass, string, ensemble, brass, reed, pipe, synth]
progList = [piano, percussion, organ, guitar, bass, string, brass, reed]#, ensemble, pipe, synth]
simplifiedProgs = {}
for vals in range(len(progList)):
    for v in progList[vals]:
        simplifiedProgs[int(v)] = int(vals)

#%%
def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

#%%
# Inversion functions
def synthPianoRoll(path, pRoll, stepDur=(0.5 / 24)):
    revProgs = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
    # Create a PrettyMIDI object
    outMidi = pm.PrettyMIDI()
    for ins in range(nbInsHandled):
        if (np.sum(pRoll[ins]) == 0):
            continue
        plt.imshow(pRoll[ins])
        curProg = pm.Instrument(program=revProgs[ins])
        # Iterate over length
        notes = np.nonzero(pRoll[ins])
        active = {}
        for t in range(len(notes[0])):
            if (active.get(notes[0][t]) is None):
                active[notes[0][t]] = []
            active[notes[0][t]].append(notes[1][t])
        for k, v in active.items():
            finalNotes = group_consecutives(v)
            for n in finalNotes:
                sTime = n[0] * stepDur
                eTime = (n[-1] * stepDur) + (stepDur - 0.01)
                # Create a Note instance, starting at 0s and ending at .5s
                note = pm.Note(velocity=int(pRoll[ins][k][n[0]]), pitch=int(k), start=sTime, end=eTime)
                # Add it to our cello instrument
                curProg.notes.append(note)
        # Add the instrument to the PrettyMIDI object
        outMidi.instruments.append(curProg)
    # Synthesize audio
    y = outMidi.fluidsynth(fs=22050)
    librosa.output.write_wav(path, y, 22050)


#%%
rollDiv = 16
eventsDiv = 24
nbBeats = 8
rollLen = rollDiv * nbBeats
eventsLen = eventsDiv * nbBeats
basePath = '/Users/esling/Datasets/symbolic/clean_midi/'
destPath = '/Users/esling/Datasets/symbolic/clean_midi/processed'
splits = ['test']


for sp in splits:
    curCounts = [0, 0, 0]
    dataPath = basePath + sp
    data_files = sorted(glob.glob(dataPath + '/*'))
    for i in range(len(data_files)):
        if os.path.splitext(data_files[i])[1] == '.mid':
            print(data_files[i])
            try:
                pmFile = pm.PrettyMIDI(data_files[i])
            except:
                continue
            instruments = pmFile.instruments
            onsets = pmFile.get_onsets()
            tempo = pmFile.estimate_tempo()
            beats = pmFile.get_beats(start_time=onsets[0])
            downbeats = pmFile.get_downbeats(start_time=onsets[0])
            # Estimate resolution
            beatLen = beats[1] - beats[0]
            downbeats = np.append(downbeats, downbeats[-1] + (downbeats[-1] - downbeats[-2]))
            fs =  16 / beatLen
            fullList = np.array([])
            for b in range(len(downbeats)-1):
                divList = np.linspace(downbeats[b], downbeats[b+1], 65)
                fullList = np.append(fullList, divList[:-1])
            pRoll = [None] * nbInsHandled
            pRollFixed = [None] * nbInsHandled
            nbMeasRoll = 0;
            nbMeasRollFixed = 0;
            destInst = [-1] * len(instruments)
            orderedList = [None] * nbInsHandled
            for o in range(nbInsHandled):
                orderedList[o] = {}
            finalList = [None] * nbInsHandled
            broke = 0
            for ins in range(len(instruments)):
                val = instruments[ins].program
                if (simplifiedProgs.get(val) is None):
                    continue
                fIns = simplifiedProgs[val]
                if (instruments[ins].is_drum):
                    fIns = 1
                destInst[ins] = fIns
                tmpRoll = instruments[ins].get_piano_roll(times=fullList)
                if (np.sum(np.isnan(tmpRoll)) > 0):
                    print('[Skipping nan file]')
                    broke = 1
                    break
                if (pRoll[fIns] is None):
                    pRoll[fIns] = tmpRoll
                    pRollFixed[fIns] = instruments[ins].get_piano_roll(fs)
                else:
                    pRoll[fIns] += tmpRoll
                    fPRoll = instruments[ins].get_piano_roll(fs)
                    if (fPRoll.shape[1] > pRollFixed[fIns].shape[1]):
                        pRollFixed[fIns] = np.pad(pRollFixed[fIns], ((0, 0), (0, fPRoll.shape[1] - pRollFixed[fIns].shape[1])), 'constant')
                        pRollFixed[fIns] += fPRoll
                    else:
                        pRollFixed[fIns][:, :fPRoll.shape[1]] += fPRoll
                    pRollFixed[fIns][pRollFixed[fIns] > 127] = 127
                nbMeasRoll = (np.floor(pRoll[fIns].shape[1] / rollLen) > nbMeasRoll) and int(np.floor(pRoll[fIns].shape[1] / rollLen)) or nbMeasRoll
                nbMeasRollFixed = (np.floor(pRoll[fIns].shape[1] / rollLen) > nbMeasRollFixed) and int(np.floor(pRoll[fIns].shape[1] / rollLen)) or nbMeasRollFixed
            if (broke == 1):
                continue
            # Export piano rolls
            for m in range(nbMeasRoll):
                data = np.zeros((nbInsHandled, 128, 128))
                for ins in range(nbInsHandled):
                    if ((pRoll[ins] is None) or np.sum(pRoll[ins]) == 0):
                        continue
                    data[ins] += pRoll[ins][:, int(m*rollLen):int((m+1)*rollLen)]
                if (np.sum(data) == 0):
                    continue
                np.savez_compressed(destPath + '/roll_sync/' + sp + '/measure_' + str(curCounts[0]), data=data)
                curCounts[0] += 1