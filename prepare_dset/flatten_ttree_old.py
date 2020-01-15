import threading as thr
from sys import argv
from queue import Queue
from ROOT import TFile, TTree, TChain
from array import array

multithread=False
max_threads = 3

start, stop = 100, 200 # Process files numbers [start, stop)


def flatten(rootfile_in, rootfile_out):
    t_in = TChain("sk2p2")
    t_in.Add(rootfile_in)
    entries = t_in.GetEntries()

    f = TFile( rootfile_out, 'recreate')
    t = TTree( 'sk2p2', 'Flattened tree with timing peaks')

    def get_low_id(r2,z):
        lowid = 5
        if r2 > 80: lowid -= 1
        if r2 > 130: lowid -= 1
        if r2 > 160: lowid -= 1
        if r2 > 190: lowid -= 1
        if r2 > 210: lowid -= 1
        if z > 12: lowid -= 1
        if z > 14: lowid -= 1
        if z > 15: lowid -= 1
        if lowid < 0: lowid = 0
        return lowid

    event_num = array('i',[0])
    is_signal = array('i',[0])
    np, N200M, T200M = array('i',[0]),array('i',[0]),array('f',[0])
    N10, N200, N10d = array('i',[0]),array('i',[0]),array('i',[0])
    Nc, Nback, N300 = array('i',[0]),array('i',[0]),array('i',[0])
    trms, trmsdiff, fpdist = array('f',[0]),array('f',[0]),array('f',[0])
    bpdist, fwall,bwall = array('f',[0]),array('f',[0]),array('f',[0])
    pvx,pvy,pvz = array('f',[0]),array('f',[0]),array('f',[0])
    bse, mintrms_3, mintrms_6 = array('f',[0]),array('f',[0]),array('f',[0])
    Q10, Qrms, Qmean = array('f',[0]),array('f',[0]),array('f',[0])
    thetarms,phirms,thetam = array('f',[0]),array('f',[0]),array('f',[0])
    dt,dtn,ratio = array('f',[0]),array('f',[0]),array('f',[0])
    nvx,nvy,nvz = array('f',[0]),array('f',[0]),array('f',[0])
    tindex,n40index,Neff = array('i',[0]),array('i',[0]),array('i',[0])

    Nc1,NhighQ,NLowtheta = array('i',[0]),array('i',[0]),array('i',[0])
    NlowQ,Nlow1,Nlow2 = array('i',[0]),array('i',[0]),array('i',[0])
    Nlow3,Nlow4,Nlow5 = array('i',[0]),array('i',[0]),array('i',[0])
    Nlow6,Nlow7,Nlow8 = array('i',[0]),array('i',[0]),array('i',[0])
    Nlow9,Nlow = array('i',[0]),array('i',[0])

    #ncomb3,cable =array('i',[0]),array('i',[0])

    # new variables
    # goodness_combined, goodness_prompt1, goodness_neutron1 = array('f',[0]),array('f',[0]),array('f',[0])
    # goodness_prompt, goodness_neutron, goodness_window = array('f',[0]),array('f',[0]),array('f',[0])
    # cvertexx, cvertexy, cvertexz = array('f',[0]),array('f',[0]),array('f',[0])
    # pvertexx, pvertexy, pvertexz = array('f',[0]),array('f',[0]),array('f',[0])
    # nvertexx, nvertexy, nvertexz = array('f',[0]),array('f',[0]),array('f',[0])
    # goodness_ratio = array('f',[0])

    t.Branch( 'event_num', event_num, 'event_num/I')
    t.Branch ( 'is_signal', is_signal, 'is_signal/I')
    t.Branch( 'np', np, 'np/I' )
    t.Branch( 'N200M', N200M, 'N200M/I' )
    t.Branch( 'T200M', T200M, 'T200M/I' )

    t.Branch( 'N10', N10, 'N10/I' )
    t.Branch( 'N200', N200, 'N200/I' )
    t.Branch( 'N10d', N10d, 'N10d/I' )
    t.Branch( 'Nc', Nc, 'Nc/I' )
    t.Branch( 'Nback', Nback, 'Nback/I' )
    t.Branch( 'N300', N300, 'N300/I' )
    t.Branch( 'trms', trms, 'trms/F' )
    t.Branch( 'trmsdiff', trmsdiff, 'trmsdiff/F' )
    t.Branch( 'fpdist', fpdist, 'fpdist/F' )
    t.Branch( 'bpdist', bpdist, 'bpdist/F' )
    t.Branch( 'fwall', fwall, 'fwall/F' )
    t.Branch( 'bwall', bwall, 'bwall/F' )
    t.Branch( 'pvx', pvx, 'pvx/F' )
    t.Branch( 'pvy', pvy, 'pvy/F' )
    t.Branch( 'pvz', pvz, 'pvz/F' )
    t.Branch( 'bse', bse, 'bse/F' )
    t.Branch( 'mintrms_3', mintrms_3, 'mintrms_3/F' )
    t.Branch( 'mintrms_6', mintrms_6, 'mintrms_6/F' )
    t.Branch( 'Q10', Q10, 'Q10/F' )
    t.Branch( 'Qrms', Qrms, 'Qrms/F' )
    t.Branch( 'Qmean', Qmean, 'Qmean/F' )
    t.Branch( 'thetarms', thetarms, 'thetarms/F' )
    t.Branch( 'NLowtheta', NLowtheta, 'NLowtheta/I' )
    t.Branch( 'phirms', phirms, 'phirms/F' )
    t.Branch( 'thetam', thetam, 'thetam/F' )
    t.Branch( 'dt', dt, 'dt/F' )
    t.Branch( 'dtn', dtn, 'dtn/F' )
    t.Branch( 'nvx', nvx, 'nvx/F' )
    t.Branch( 'nvy', nvy, 'nvy/F' )
    t.Branch( 'nvz', nvz, 'nvz/F' )
    t.Branch( 'tindex', tindex, 'tindex/I' )
    t.Branch( 'n40index', n40index, 'n40index/I' )
    t.Branch( 'Neff', Neff, 'Neff/I' )
    t.Branch( 'ratio', ratio, 'ratio/F' )
    t.Branch( 'Nc1', Nc1, 'Nc1/I' )
    t.Branch( 'NhighQ', NhighQ, 'NhighQ/I' )
    t.Branch( 'NlowQ', NlowQ, 'NlowQ/I' )
    t.Branch( 'Nlow1', Nlow1, 'Nlow1/I' )
    t.Branch( 'Nlow2', Nlow2, 'Nlow2/I' )
    t.Branch( 'Nlow3', Nlow3, 'Nlow3/I' )
    t.Branch( 'Nlow4', Nlow4, 'Nlow4/I' )
    t.Branch( 'Nlow5', Nlow5, 'Nlow5/I' )
    t.Branch( 'Nlow6', Nlow6, 'Nlow6/I' )
    t.Branch( 'Nlow7', Nlow7, 'Nlow7/I' )
    t.Branch( 'Nlow8', Nlow8, 'Nlow8/I' )
    t.Branch( 'Nlow9', Nlow9, 'Nlow9/I' )
    t.Branch( 'Nlow', Nlow, 'Nlow/I' )
    #t.Branch( 'ncomb3', ncomb3, 'ncomb3/I' )
    #t.Branch( 'cable', cable, 'cable/I' )

    # new variables
    # t.Branch( 'goodness_combined', goodness_combined, 'goodness_combined/F' )
    # t.Branch( 'goodness_prompt1', goodness_prompt1, 'goodness_prompt1/F' )
    # t.Branch( 'goodness_neutron1', goodness_neutron1, 'goodness_neutron1/F' )
    # t.Branch( 'goodness_prompt', goodness_prompt, 'goodness_prompt/F' )
    # t.Branch( 'goodness_neutron', goodness_neutron, 'goodness_neutron/F' )
    # t.Branch( 'goodness_window', goodness_window, 'goodness_window/F' )
    # t.Branch( 'pvertexx', pvertexx, 'pvertexx/F' )
    # t.Branch( 'pvertexy', pvertexy, 'pvertexy/F' )
    # t.Branch( 'pvertexz', pvertexz, 'pvertexz/F' )
    # t.Branch( 'cvertexx', cvertexx, 'cvertexx/F' )
    # t.Branch( 'cvertexy', cvertexy, 'cvertexy/F' )
    # t.Branch( 'cvertexz', cvertexz, 'cvertexz/F' )
    # t.Branch( 'nvertexx', nvertexx, 'nvertexx/F' )
    # t.Branch( 'nvertexy', nvertexy, 'nvertexy/F' )
    # t.Branch( 'nvertexz', nvertexz, 'nvertexz/F' )
    # t.Branch( 'goodness_ratio', goodness_ratio, 'goodness_ratio/F' )

    for entry in range(entries):
        t_in.GetEntry(entry)
        event_num[0]=entry
        np[0], N200M[0], T200M[0] = t_in.np, t_in.N200M, t_in.T200M

        #if entry%10000==0: print("%d/%d entries read"%(entry,entries))
        for p in range(np[0]):
            N10[0], N200[0], N10d[0] = t_in.N10[p], t_in.N200[p], t_in.N10d[p]
            Nc[0], Nback[0], N300[0] = t_in.Nc[p], t_in.Nback[p], t_in.N300[p]
            trms[0], trmsdiff, fpdist[0] = t_in.trms[p], t_in.trmsdiff[p], t_in.fpdist[p]
            bpdist[0], fwall[0],bwall[0] = t_in.bpdist[p], t_in.fwall[p],t_in.bwall[p]
            pvx[0],pvy[0],pvz[0] = t_in.pvx[p],t_in.pvy[p],t_in.pvz[p]
            bse[0], mintrms_3[0], t_in.mintrms_6[0] = t_in.bse[p], t_in.mintrms_3[p], t_in.mintrms_6[p]
            Q10[0], Qrms[0], Qmean[0] = t_in.Q10[p], t_in.Qrms[p], t_in.Qmean[p]
            thetarms[0],phirms[0],thetam[0] = t_in.thetarms[p],t_in.phirms[p],t_in.thetam[p]
            dt[0],dtn[0],ratio[0] = t_in.dt[p],t_in.dtn[p],t_in.ratio[p]
            nvx[0],nvy[0],nvz[0] = t_in.nvx[p],t_in.nvy[p],t_in.nvz[p]
            tindex[0],n40index[0],Neff[0] = t_in.tindex[p],t_in.n40index[p],t_in.Neff[p]

            Nc1[0],NhighQ[0],NLowtheta[0] = t_in.Nc1[p],t_in.NhighQ[p],t_in.NLowtheta[p]
            NlowQ[0],Nlow1[0],Nlow2[0] = t_in.NlowQ[p],t_in.Nlow1[p],t_in.Nlow2[p]
            Nlow3[0],Nlow4[0],Nlow5[0] = t_in.Nlow3[p],t_in.Nlow4[p],t_in.Nlow5[p]
            Nlow6[0],Nlow7[0],Nlow8[0] = t_in.Nlow6[p],t_in.Nlow7[p],t_in.Nlow8[p]
            Nlow9[0] = t_in.Nlow9[p]

            # new variables
            # goodness_combined[0], goodness_prompt1[0], goodness_neutron1[0] = t_in.goodness_combined[p], t_in.goodness_prompt1[p], t_in.goodness_neutron1[p]
            # goodness_prompt[0], goodness_neutron[0], goodness_window[0] = t_in.goodness_prompt[p], t_in.goodness_neutron[p], t_in.goodness_window[p]
            # cvertexx[0], cvertexy[0], cvertexz[0] = t_in.cvertexx[p], t_in.cvertexy[p], t_in.cvertexz[p]
            # nvertexx[0], nvertexy[0], nvertexz[0] = t_in.nvertexx[p], t_in.nvertexy[p], t_in.nvertexz[p]
            # pvertexx[0], pvertexy[0], pvertexz[0] = t_in.pvertexx[p], t_in.pvertexy[p], t_in.pvertexz[p]

            # goodness_ratio[0] = goodness_combined[0] - goodness_neutron[0] - goodness_prompt[0]

            r2 = pvx[0]**2 + pvy[0]**2
            nlows = [Nlow1[0],Nlow2[0],Nlow3[0],Nlow4[0],Nlow5[0],Nlow6[0],Nlow7[0],Nlow8[0],Nlow9[0]]
            nlowid = get_low_id(r2,pvz[0])
            Nlow[0] = nlows[nlowid]

            is_signal[0] = abs(dt[0] - 200.9e3) < 200

            t.Fill()

    f.Write()
    f.Close()
    print("Flattened file: ", rootfile_in)

def worker(q):
    while not q.empty():
        rin, rout = q.get()
        flatten(rin, rout)
        q.task_done

if __name__=='__main__':

    try:
        start, stop = int(argv[1]), int(argv[2])
    except IndexError:
        pass
    print("Flattening files from number %i to %i" % (start, stop-1))

    rootfile_ins = ["/data_CMS/cms/giampaolo/ntag-dset/root/shift0/%03i.root"%i for i in range(start,stop)]
    rootfile_outs = ["/data_CMS/cms/giampaolo/ntag-dset/root_flat/shift0/%03i.root"%i for i in range(start,stop)]

    if multithread:
        myqueue = Queue()
        for e in zip(rootfile_ins, rootfile_outs):
            myqueue.put(e)

        for i in range(max_threads):
            mythread = thr.Thread(target=worker, args=(myqueue,))
            mythread.start()

        print("Waiting for", myqueue.qsize(), "tasks")
        myqueue.join()
        print("All done!")

    else:
        for rin,rout in zip(rootfile_ins,rootfile_outs):
            flatten(rin, rout)