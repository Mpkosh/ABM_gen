import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import seir_discrete
from sklearn.metrics import root_mean_squared_error as rmse


def cpoint_perc_people(seed_df, perc=0.01, popul=200000):
    the_day = seed_df[seed_df.I_H1N1 >= popul*perc].index
    if the_day.shape[0]:
        return the_day[0]
    else:
        return -1


def constant_betas_all(days_before=10, sigma=1/2, gamma=1/6,
                      folder = 'sampled_200k_res_recalc', n_seeds=30):
    plot = True
    
    cols = 2
    rows = (n_seeds + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5*rows))
    axes = axes.flatten()
    
    pt_1, ph_1 = [], []
    pt_2, ph_2 = [], []
    
    for i, seed in enumerate(np.arange(n_seeds)):
        '''
        if folder != 'sampled_200k_res_recalc':
            df_b = pd.read_csv(f'{folder}/seirb_seed_{seed}.csv',
                              sep='\t')
        else:
        '''
        df_b = pd.read_csv(f'{folder}/seirb_seed_{seed}.csv')
        
        fin = df_b.shape[0]
        real_peakt = df_b.I_H1N1.argmax()
        real_peakh = df_b.I_H1N1.max()

        st_day = real_peakt - days_before
        ax = axes[i]
        ax_beta = ax.twinx()
        
        if st_day>0:
            betas = df_b.beta_H1N1

            y0 = df_b.iloc[st_day,:4].values
            ts = np.arange(fin-st_day)

            col = 'tab:red'
            lab = 'SEIR I (exp.mean beta)'

            res_dict = dict()

            for b in np.arange(0.0, 2e-5, 2e-7):

                #ax_beta.plot(b, ls='--', color='tab:green')
                r = seir_discrete.seir_model(y0, ts, b, 
                                                   sigma, gamma, stype='d', 
                                                   beta_t=False).T
                S,E,I,R = r
                if plot:
                    ax.plot(ts+st_day, I, color=col, ls='-', alpha=0.3)
                predicted_peakt = I.argmax()+st_day
                predicted_peakh = I.max()
                
                rmse_pt = rmse([real_peakt], [predicted_peakt])
                rmse_ph = rmse([real_peakh], [predicted_peakh])
                res_dict[b] = [rmse_pt,rmse_ph]

            best_b = pd.DataFrame.from_dict(res_dict).T.reset_index()
            best_b.columns = ['beta', 'rmse_pt', 'rmse_ph']

            best_b_peak_time = best_b.nsmallest(1,'rmse_pt').iloc[0,0]   
            S,E,Ipt,R = seir_discrete.seir_model(y0, ts, best_b_peak_time, sigma, gamma, stype='d', 
                                               beta_t=False).T

            best_b_peak_h = best_b.nsmallest(1,'rmse_ph').iloc[0,0]
            S,E,Iph,R = seir_discrete.seir_model(y0, ts, best_b_peak_h, sigma, gamma, stype='d', 
                                               beta_t=False).T
            
            predicted_peakt = Ipt.argmax()+st_day
            predicted_peakh = Ipt.max()
            pt_1.append(predicted_peakt-real_peakt)
            ph_1.append(predicted_peakh-real_peakh)
            
            predicted_peakt = Iph.argmax()+st_day
            predicted_peakh = Iph.max()
            pt_2.append(predicted_peakt-real_peakt)
            ph_2.append(predicted_peakh-real_peakh)
                
            if plot:
                ax.plot(df_b.I_H1N1, color='tab:blue', marker='.', ls='')

                ax.plot(ts+st_day, Ipt, color='yellow', ls='-', alpha=1, lw=2,
                    label=f'I; best beta (peaktime) {best_b_peak_time:.7f}') 

                ax_beta.plot(betas, color='gray', alpha=0.3, label='beta from df')
                ax.axvline(st_day, ls=':', color='red')
                ax.plot(ts+st_day, Iph, color='lime', ls='-', alpha=1, 
                        label=f'I; best beta (peakheight) {best_b_peak_h:.7f}')   


                ax.set_title(f'Seed {seed}, switch on day {st_day},\n'+\
                             f'peak time {real_peakt}, predicted peak time {predicted_peakt}')
                ax.legend()

                ax_beta.axhline(best_b_peak_time, color='gold', ls=':', alpha=0.5)
                ax_beta.axhline(best_b_peak_h, color='lime', ls=':', alpha=1)
                ax.grid()
                
                # обрезаем график
                last = df_b[(df_b.E_H1N1==0)&(df_b.I_H1N1==0)].index
                if last.shape[0]:
                    ax.set_xlim(-2, last[0]+10)
                
    fig.tight_layout()
    plot_peaks_area(pts = [pt_2], 
                    phs = [ph_2],
                    labels=['best_by_ph'])
    
    
def constant_betas(seed=0, days_before=10, sigma=1/2, gamma=1/6,
                  plot=True, folder = 'sampled_200k_res_recalc'):

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax_beta = ax.twinx()
    df_b = pd.read_csv(f'{folder}/seirb_seed_{seed}.csv')

    fin = df_b.shape[0]
    real_peakt = df_b.I_H1N1.argmax()
    real_peakh = df_b.I_H1N1.max()
    
    st_day = real_peakt - days_before
    
    if st_day>0:
        betas = df_b.beta_H1N1

        y0 = df_b.iloc[st_day,:4].values
        ts = np.arange(fin-st_day)

        col = 'tab:red'
        lab = 'SEIR I (exp.mean beta)'

        res_dict = dict()
        Inf = df_b.iloc[st_day,2]

        for b in np.arange(0.0, 2e-5, 2e-7):

            #ax_beta.plot(b, ls='--', color='tab:green')
            S,E,I,R = seir_discrete.seir_model(y0, ts, b, 
                                               sigma, gamma, stype='d', 
                                               beta_t=False).T
            if plot:
                ax.plot(ts+st_day, I, color=col, ls='-', alpha=0.3)
            predicted_peakt = I.argmax()+st_day
            predicted_peakh = I.max()

            rmse_pt = rmse([real_peakt], [predicted_peakt])
            rmse_ph = rmse([real_peakh], [predicted_peakh])
            res_dict[b] = [rmse_pt,rmse_ph]
        
        best_b = pd.DataFrame.from_dict(res_dict).T.reset_index()
        best_b.columns = ['beta', 'rmse_pt', 'rmse_ph']
        
        best_b_peak_time = best_b.nsmallest(1,'rmse_pt').iloc[0,0]   
        S,E,Ipt,R = seir_discrete.seir_model(y0, ts, best_b_peak_time, sigma, gamma, stype='d', 
                                           beta_t=False).T

        best_b_peak_h = best_b.nsmallest(1,'rmse_ph').iloc[0,0]
        S,E,Iph,R = seir_discrete.seir_model(y0, ts, best_b_peak_h, sigma, gamma, stype='d', 
                                           beta_t=False).T
        if plot:
            ax.plot(df_b.I_H1N1, color='tab:blue', marker='.', ls='')
            
            ax.plot(ts+st_day, Ipt, color='yellow', ls='-', alpha=1, lw=2,
                label=f'I; best beta (peaktime) {best_b_peak_time:.7f}') 
            
            ax_beta.plot(betas, color='gray', alpha=0.3, label='beta from df')
            ax.axvline(st_day, ls=':', color='red')
            ax.plot(ts+st_day, Iph, color='lime', ls='-', alpha=1, 
                    label=f'I; best beta (peakheight) {best_b_peak_h:.7f}')   


            ax.set_title(f'Seed {seed}, switch on day {st_day},\n'+\
                         f'peak time {real_peakt}, predicted peak time {predicted_peakt}')
            ax.legend()

            ax_beta.axhline(best_b_peak_time, color='gold', ls=':', alpha=0.5)
            ax_beta.axhline(best_b_peak_h, color='lime', ls=':', alpha=0.5)
            ax.grid()
            
            
            
            
    return [betas, real_peakt, Inf, best_b, best_b_peak_time, best_b_peak_h]

    
def plot_hybrid(switch_method='frac_people', perc=0.01, 
                sigma=1/2, gamma=1/6, plot_traj=True):
    
    folder = 'sampled_200k_res_recalc'
    if plot_traj:
        fig, axes = plt.subplots(15, 2, figsize=(12, 35))
        axes = axes.flatten()

    pt_1, ph_1 = [], []
    pt_2, ph_2 = [], []
    pt_3, ph_3 = [], []
    chosen_st = []
    real_peaks = []
    
    for i, seed in enumerate(np.arange(0,30)):
        
        df_b = pd.read_csv(f'{folder}/seirb_seed_{seed}.csv')
        #inc = pd.read_csv(f'{folder}/incidence_seed_{i}.csv', sep='\t')['H1N1']
        
        fin = 250
        
        # выбираем способ переключения
        if switch_method=='frac_people':
            st_day = cpoint_perc_people(df_b, perc=perc)
        else:
            st_day = switch_method
            
        
        if st_day > 0:
            real_peakt = df_b.I_H1N1.argmax()
            real_peakh = df_b.I_H1N1.max()
            real_peaks.append(real_peakt)
            chosen_st.append(st_day)
            
            betas = df_b.beta_H1N1[:st_day]
            beta_exp = betas[:st_day].expanding().mean()
            beta_roll = betas[:st_day].rolling(7).mean()
            
            if plot_traj:
                ax = axes[i]
                ax.axvline(st_day, ls=':', color='red')
                ax.plot(df_b.I_H1N1, label='I', color='tab:blue', marker='.', ls='-', lw=0.5)

                ax_beta = ax.twinx()
            
                ax_beta.plot(betas[:st_day], color='gray', alpha=0.4, label='beta from df')

                ax_beta.plot(beta_exp, color='gray',  label='beta exp', alpha=0.7)
                ax_beta.plot(beta_roll, color='gray',  label='beta roll', alpha=0.7, ls='--')
            
            # ____ после дня переключения, expanding mean
            y0 = df_b.iloc[st_day,:4].values
            ts = np.arange(fin-st_day)
            
            for beta_val, pt_l, ph_l, col, lab in zip([beta_exp.values[-1], 
                                                         beta_roll.values[-1],
                                                         betas.values[-1]],
                                                       [pt_1,pt_2,pt_3], [ph_1,ph_2,ph_3],
                                                       ['tab:red','tab:orange','tab:green'],
                                                       ['SEIR I (exp.mean beta)',
                                                        'SEIR I (roll.mean beta)',
                                                        'SEIR I (last val beta)']):
                S,E,I,R = seir_discrete.seir_model(y0, ts, beta_val, 
                                                   sigma, gamma, stype='d', 
                                                   beta_t=False).T
                
                if plot_traj:
                    ax.plot(ts+st_day, I, color=col, ls='--', alpha=0.8, label=lab)
                    '''
                    seir_inc = np.diff(I)
                    seir_inc[seir_inc<0] = 0
                    ax.plot(ts[1:]+st_day, seir_inc, color=col, ls='-', alpha=0.8, label='predicted I diff')
                    '''
            
                predicted_peakt = I.argmax()+st_day
                predicted_peakh = I.max()
                pt_l.append(predicted_peakt-real_peakt)
                ph_l.append(predicted_peakh-real_peakh)
                
            if plot_traj:    
                ax.legend()    
                ax.grid()
                ax.set_title(f'Seed {seed}, switch on day {st_day}')
    
    if plot_traj:
        fig.tight_layout()
    
    plot_peaks_area(pts = [pt_1,pt_2,pt_3], 
                    phs = [ph_1,ph_2,ph_3],
                    labels=['expanding','rolling','real'])
    
    return [[pt_1,pt_2,pt_3], [ph_1,ph_2,ph_3], chosen_st, real_peaks]
    

def plot_hybrid_3(switch_method='frac_people', perc=0.01, 
                beta_type='expanding', sigma=1/2, gamma=1/6):
    folder = 'sampled_200k_res_recalc'
    fig, axes = plt.subplots(15, 2, figsize=(12, 35))
    axes = axes.flatten()

    pt_1, ph_1 = [], []
    pt_2, ph_2 = [], []
    pt_3, ph_3 = [], []
    
    
    for i, seed in enumerate(np.arange(0,30)):
        ax = axes[i]
        df_b = pd.read_csv(f'{folder}/seirb_seed_{seed}.csv')

        fin = 250
        
        # выбираем способ переключения
        if switch_method=='frac_people':
            st_day = cpoint_perc_people(df_b, perc=perc)
        else:
            st_day = df_b.I_H1N1.argmax() - switch_method

        if st_day > 0:
            real_peakt = df_b.I_H1N1.argmax()
            real_peakh = df_b.I_H1N1.max()
            
            
            ax.axvline(st_day, ls=':', color='red')
            ax.plot(df_b.I_H1N1, label='I', color='tab:blue', marker='.')

            ax_beta = ax.twinx()
            betas = df_b.beta_H1N1[:st_day]
            ax_beta.plot(betas[:st_day], color='gray', alpha=0.4, label='beta from df')
            
            if beta_type == 'expanding':
                beta_smooth = betas[:st_day].expanding().mean()
            elif beta_type == 'rolling':
                beta_smooth = betas[:st_day].rolling(7).mean()
            elif beta_type == 'real':
                beta_smooth = betas[:st_day]
            ax_beta.plot(beta_smooth, color='gray',  label='beta for input', alpha=0.7)
            
            # ____ после дня переключения, expanding mean
            y0 = df_b.iloc[st_day,:4].values
            ts = np.arange(fin-st_day)

            S,E,I,R = seir_discrete.seir_model(y0, ts, beta_smooth.values[-1], sigma, gamma, 
                                 stype='d', beta_t=False).T
            ax.plot(ts+st_day, I, color='tab:red', ls='--', alpha=0.8, 
                    label='SEIR I (from switch, smooth beta)')
            
            predicted_peakt = I.argmax()+st_day
            predicted_peakh = I.max()
            pt_1.append(predicted_peakt-real_peakt)
            ph_1.append(predicted_peakh-real_peakh)

            ts = np.arange(fin)
            y0 = df_b.iloc[0,:4].values
            
            
            # ____ до и после дня переключения -- на основе expanding mean
            expmean = beta_smooth.iloc[:st_day].values
            S,E,I,R = seir_discrete.seir_model(y0, ts, 
                                               [*expmean,*[beta_smooth.iloc[-1]]*(fin-st_day)], 
                                               sigma, gamma, stype='d', beta_t=True).T
            ax.plot(np.arange(fin), I, color='tab:orange', ls='--', alpha=0.8, 
                    label='SEIR I (from day 0, smooth beta)')
            predicted_peakt = I.argmax()
            predicted_peakh = I.max()
            pt_2.append(predicted_peakt-real_peakt)
            ph_2.append(predicted_peakh-real_peakh)

            
            # ____ до дня переключения реальная бета, после -- expanding mean
            expmean = betas.iloc[:st_day].values
            S,E,I,R = seir_discrete.seir_model(y0, ts, 
                                               [*expmean,*[beta_smooth.iloc[-1]]*(fin-st_day)], 
                                               sigma, gamma, stype='d', beta_t=True).T
            ax.plot(np.arange(fin), I, color='tab:green', ls='--', alpha=0.8, 
                    label='SEIR I (from day 0, beta)')
            predicted_peakt = I.argmax()
            predicted_peakh = I.max()
            pt_3.append(predicted_peakt-real_peakt)
            ph_3.append(predicted_peakh-real_peakh)

            ax.legend()    
            ax.grid()

            ax.set_title(f'Seed {seed}, switch on day {st_day}')

    fig.tight_layout()
    
    plot_peaks_area(pts = [pt_1,pt_2,pt_3], 
                    phs = [ph_1,ph_2,ph_3])
    return [[pt_1,pt_2,pt_3], [ph_1,ph_2,ph_3]]


def plot_peaks_area(pts, phs, labels=['SEIR I (from switch, smooth beta)',
                                         'SEIR I (from day 0, smooth beta)',
                                         'SEIR I (from day 0, beta)'],
                   colors = ['tab:red','tab:orange','tab:green']):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.axhline(0, lw=1.5, ls=':')
    ax.axvline(0, lw=1.5, ls=':')
    
    for pt_val, ph_val, color, label in zip(pts, phs, colors, labels):
        pt = pd.Series(pt_val)
        ph = pd.Series(ph_val)
        ax.scatter(pt, ph, color=color, alpha = 0.6, label=label)

        hull = ConvexHull(pd.concat([pt,ph], axis=1))
        ax.fill(pt.iloc[hull.vertices], 
                     ph.iloc[hull.vertices], alpha=0.2, color=color)
        
    ax.grid()
    ax.set_xlabel('predicted - real time')
    ax.set_ylabel('predicted - real height')
    ax.set_title('peak metrics for beta')
    ax.legend()