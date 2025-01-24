# %% Libraries
import os  # permette di gestire i percorsi in modo indipendente dal sistema operativo
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tqdm
import streamlit as st  # servirà nel caso di sviluppo di una webapp
# from io import BytesIO # servirà nel caso si voglia generare un file excel dal datarame per poterlo scaricare dal una webapp strealit
# import openai # servirà nel caso di applicazioni AI
# %% Funzione generale previsore


def df_forecast(db_group, k, alpha, r, P, grafico, scenario, serie):
    Y = db_group.get_group(k)
    x = np.array(Y[serie])

    # step 1: Applicare lo smoothing esponenziale per calcolare il valore di equilibrio
    if np.var(x) > 0:
        model_ses = sm.tsa.SimpleExpSmoothing(x)
        fitted_ses = model_ses.fit(
            smoothing_level=alpha, optimized=False)
        fitted_ses = fitted_ses.fittedvalues
    else:
        fitted_ses = [np.mean(x)] * len(x)

    # step 2: Modello di Holt con trend lineare per generare previsioni
    if np.var(x) > 0:
        model_holt = sm.tsa.ExponentialSmoothing(
            x, trend="add", seasonal=None)
        fitted_holt = model_holt.fit()
        forecast_holt = fitted_holt.forecast(steps=P)
    else:
        forecast_holt = [np.mean(x)] * P

    equilibrium_value = fitted_ses[-1]*(1+scenario)
    # step 3:  Modificare le previsioni per farle convergere all'equilibrio

    forecast_converged = []
    for i, val in enumerate(forecast_holt):
        weight = np.exp(-r * i)  # Peso esponenziale
        adjusted_val = weight * val + \
            (1 - weight) * equilibrium_value
        forecast_converged.append(adjusted_val)
    # la serie finale con le previsioni a P anni

    # step 4: risultato e completamento del dataframe
    xf = forecast_converged  # la serie predetta
    xf = [max(0, val) for val in xf]
    xt = []  # gli anni di previsione
    for t in range(P):
        xt.append(Y['TIME'].iloc[-1]+t+1)

    if grafico == "Sì":
        plt.plot(x, label="Serie originale", marker='o')
        plt.plot(fitted_ses,
                 label="Smoothing Esponenziale", linestyle='--')
        plt.plot(range(len(x), len(x) + len(forecast_holt)),
                 forecast_holt, label="Forecast Holt", linestyle='--')
        plt.plot(range(len(x), len(x) + len(forecast_converged)),
                 xf, label="Convergenza all'equilibrio", linestyle=':')
        plt.axhline(y=equilibrium_value, color='gray',
                    linestyle='--', label="Valore di equilibrio")
        # plt.legend()
        plt.title(
            "Forecast con convergenza all'equilibrio serie " + k, fontsize=8)
    num_records = len(xf)
    # Estrai i parametri dimensionali
    dimensional_params = Y.iloc[0].drop(
        [serie, 'TIME']).to_dict()
    new_data = {key: [value] * num_records for key,
                value in dimensional_params.items()}  # Replica i parametri
    new_data[serie] = xf  # Popola il campo serie con xf
    new_data['TIME'] = xt    # Popola il campo "Time" con xt

    Yf = pd.DataFrame(new_data)
    # step 5: accoda al dataframe originale
    if grafico == "Sì":
        return plt
    else:
        return Yf


# %%
if "dataframes" not in st.session_state:
    st.session_state["dataframes"] = {}
# %%
with st.sidebar:
    st.header(':blue[DemoSi updater]', divider='blue')
    file_selezionato = st.selectbox("Dataframe salvati nella sessione", options=list(
        st.session_state["dataframes"].keys()))
    if file_selezionato:
        dbf_csv = st.session_state["dataframes"][file_selezionato].to_csv(
            sep="|", index=False)
        st.download_button('Scarica il file - ' + file_selezionato + ' - in formato CSV',
                           data=dbf_csv, file_name=file_selezionato+".csv")
# %%

st.title("Modulo 2: Aggiornamento movimento migratorio")

# %%
if "db_pop_classe_final" in st.session_state["dataframes"]:
    st.success(
        "Il database sulla popolazione straniera e italiana per classe di età è disponibile, puoi quindi procedere con l'esecuzione del modulo")
else:
    st.error(
        "Per eseguire questo modulo è necessario costruire il database della popolazione straniera e italiana per classe di età")
    Input_file_2 = st.file_uploader(
        'Inizia caricando il file sulla popolazione straniera per classe di età ISTAT_Pop_str.csv, verifica che non vi siano cambiamenti nel nome dei campi e nel separatore')
    if Input_file_2:
        try:
            db_pop_str = pd.read_csv(
                Input_file_2, sep=',', encoding='utf-8')
            remove = ['TIPO_DATO15',  'SEXISTAT1', 'Età',
                      'Seleziona periodo', 'Flag Codes', 'Flags']
            db_pop_str.drop(columns=remove, inplace=True)
            # esclude il totale
            Sesso = ['totale']
            db_pop_str = db_pop_str[~db_pop_str['Sesso'].isin(Sesso)]
        except:
            st.error('File errato o formato file non supportato')
        # % definire le classi di età
        classi_eta = {
            'Y_UN17': ['Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8',
                       'Y9', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17'],
            'Y18-39': ['Y18', 'Y19', 'Y20', 'Y21', 'Y22', 'Y23', 'Y24', 'Y25',
                       'Y26', 'Y27', 'Y28', 'Y29', 'Y30', 'Y31', 'Y32', 'Y33',
                       'Y34', 'Y35', 'Y36', 'Y37', 'Y38', 'Y39'],
            'Y40-64': ['Y40', 'Y41', 'Y42', 'Y43', 'Y44', 'Y45', 'Y46',
                       'Y47', 'Y48', 'Y49', 'Y50', 'Y51', 'Y52', 'Y53',
                       'Y54', 'Y55', 'Y56', 'Y57', 'Y58', 'Y58', 'Y59', 'Y60', 'Y61',
                       'Y62', 'Y63', 'Y64'],
            'Y_GE65': ['Y65', 'Y66', 'Y67', 'Y68', 'Y69', 'Y70', 'Y71', 'Y72',
                       'Y73', 'Y74', 'Y75', 'Y76', 'Y77', 'Y78', 'Y79', 'Y80',
                       'Y81', 'Y82', 'Y83', 'Y84', 'Y85', 'Y86', 'Y87', 'Y88',
                       'Y89', 'Y90', 'Y91', 'Y92', 'Y93', 'Y94', 'Y95', 'Y96',
                       'Y97', 'Y98', 'Y99', 'Y100', 'Y_GE100']
        }

        # Lista per accumulare i record
        records = []

        # Raggruppa per le colonne chiave
        group_ps = db_pop_str.groupby(
            ['ITTER107', 'Territorio', 'Tipo di indicatore demografico', 'Sesso', 'TIME'])

        for key, current_group_p in group_ps:
            for eta_label, eta_values in classi_eta.items():
                # Filtra le righe dove ETA1 è nella lista della classe corrente
                filtered_group = current_group_p[current_group_p['ETA1'].isin(
                    eta_values)]

                # Calcola la somma dei valori
                pop_cls = filtered_group['Value'].sum()

                # Crea un dizionario con i dati del nuovo record
                record = {
                    'ITTER107': key[0],
                    'Territorio': key[1],
                    'Tipo di indicatore demografico': key[2],
                    'ISO': "FOR",
                    'Sesso': key[3],
                    'ETA1': eta_label,  # Imposta ETA1 al nome della classe
                    'TIME': key[4],
                    'Value': pop_cls,  # Inserisce il valore calcolato
                    'Unicode': f"{key[0]}FOR{key[3]}{eta_label}"
                }

                # Aggiungi il record alla lista
                records.append(record)
        # Converte la lista di record in un DataFrame
        db_pop_classe_str = pd.DataFrame(records)
        st.write(
            "Ecco un estratto della popolazione stranierà per classe di età")
        st.write(db_pop_classe_str.head(5))

        Input_file_3 = st.file_uploader(
            'Prosegui cricando il file sulla popolazione totale per classe di età ISTAT_Pop_tot.csv, verifica che non vi siano cambiamenti nel nome dei campi e nel separatore')
        if Input_file_3:
            try:
                db_pop_tot = pd.read_csv(
                    Input_file_3, sep=',', encoding='utf-8')
                remove = ['TIPO_DATO15',  'SEXISTAT1', 'Età',
                          'Seleziona periodo', 'Flag Codes', 'Flags', 'Stato civile', 'STATCIV2']
                db_pop_tot.drop(columns=remove, inplace=True)
            except:
                st.error('File errato o formato file non supportato')
            # Lista per accumulare i record
            records = []
            # Raggruppa per le colonne chiave
            group_pt = db_pop_tot.groupby(
                ['ITTER107', 'Territorio', 'Tipo di indicatore demografico', 'Sesso', 'TIME'])
            for key, current_group_p in group_pt:
                for eta_label, eta_values in classi_eta.items():
                    # Filtra le righe dove ETA1 è nella lista della classe corrente
                    filtered_group = current_group_p[current_group_p['ETA1'].isin(
                        eta_values)]

                    # Calcola la somma dei valori
                    pop_cls = filtered_group['Value'].sum()

                    # Crea un dizionario con i dati del nuovo record
                    record = {
                        'ITTER107': key[0],
                        'Territorio': key[1],
                        'Tipo di indicatore demografico': key[2],
                        'ISO': "TOT",
                        'Sesso': key[3],
                        'ETA1': eta_label,  # Imposta ETA1 al nome della classe
                        'TIME': key[4],
                        'Value': pop_cls,  # Inserisce il valore calcolato
                        'Unicode': f"{key[0]}TOT{key[3]}{eta_label}"
                    }

                    # Aggiungi il record alla lista
                    records.append(record)
            # Converte la lista di record in un DataFrame
            db_pop_classe_tot = pd.DataFrame(records)
            # % il database per classe di età per la popolazione italiana definito per differenza
            group_ps_classe = db_pop_classe_str.groupby(
                ['ITTER107', 'Territorio', 'Tipo di indicatore demografico', 'Sesso', 'ETA1', 'TIME'])
            group_pt_classe = db_pop_classe_tot.groupby(
                ['ITTER107', 'Territorio', 'Tipo di indicatore demografico', 'Sesso', 'ETA1', 'TIME'])
            records = []
            for key, current_group_ps in group_ps_classe:
                current_group_pt = group_pt_classe.get_group(key)

                record = {
                    'ITTER107': key[0],
                    'Territorio': key[1],
                    'Tipo di indicatore demografico': key[2],
                    'ISO': "IT",
                    'Sesso': key[3],
                    'ETA1': key[4],
                    'TIME': key[5],
                    'Value': current_group_pt['Value'].iloc[0]-current_group_ps['Value'].iloc[0],
                    'Unicode': f"{key[0]}IT{key[3]}{key[4]}"
                }
                # Aggiungi il record alla lista
                records.append(record)
            db_pop_classe_it = pd.DataFrame(records)
            db_pop_classe_final = pd.concat(
                [db_pop_classe_it, db_pop_classe_str], ignore_index=True)
            st.session_state['dataframes']['db_pop_classe_final'] = db_pop_classe_final
            st.success(
                "Elaborazioni completate. Ecco un estratto della popolazione italiana e straniera per classe di età")
            st.write(db_pop_classe_final.head(5))

            try:
                del db_pop_classe_str, db_pop_classe_it, db_pop_classe_tot,    \
                    group_ps, group_ps_classe, group_pt, group_pt_classe, \
                    records, record, \
                    current_group_p, current_group_ps, current_group_pt
            except:
                pass

    # %%
if "db_pop_classe_final" in st.session_state["dataframes"]:
    scelta = st.radio(
        "Prosegui selezionando un'opzione:",
        options=["Movimenti con l'estero", "Movimenti interni"]
    )
    # %% load file
    if scelta == "Movimenti con l'estero":
        Input_file = st.file_uploader(
            'Carica il file dei cancellati ISTAT_Migrazioni_2.csv, verifica che non vi siano cambiamenti nel nome dei campi e nel separatore')
        if Input_file:
            try:
                db_cancellati = pd.read_csv(
                    Input_file, sep=',', encoding='utf-8')
                remove = ['TIPO_DATO15', 'Tipo di trasferimento', 'Paese di cittadinanza', 'SEXISTAT1', 'Età',
                          'PAESI_B', 'Seleziona periodo', 'Flag Codes', 'Flags', 'Stato estero di  destinazione']
                db_cancellati.drop(columns=remove, inplace=True)
                # Creazione codice univoco 'Unicode' e raggruppamento dei dati provinciali
                Destinazione = ['FREIGN']
                Nazionalità = ['TOTAL']
                Sesso = ['totale']
                Età = ['TOTAL']
                db_cancellati = db_cancellati[db_cancellati['PROV_DEST_Z'].isin(
                    Destinazione)]  # solo destinazione Estero
                db_cancellati = db_cancellati[~db_cancellati['ISO'].isin(
                    Nazionalità)]  # esclude il totale
                db_cancellati = db_cancellati[~db_cancellati['Sesso'].isin(
                    Sesso)]  # esclude il totale
                db_cancellati = db_cancellati[~db_cancellati['ETA1'].isin(
                    Età)]  # esclude il totale
                # definizione indici
                db_cancellati['Unicode'] = db_cancellati['ITTER107_A'] + \
                    db_cancellati['ISO']+db_cancellati['Sesso'] + \
                    db_cancellati['ETA1']
                db_cancellati = db_cancellati.sort_values(
                    by=['Unicode', 'TIME'])  # ordinamento
                st.write('Ecco un estratto del file che hai caricato')
                st.write(db_cancellati.head(5))
            except:
                st.error('File errato o formato file non supportato')
                # %

            # %% definizione dei tassi di migratorietà
            group_c = db_cancellati.groupby(['Unicode', 'TIME'])
            db_pop_classe_final = st.session_state['dataframes']['db_pop_classe_final']
            group_p = db_pop_classe_final.groupby(['Unicode', 'TIME'])
            # %
            db_cancellati_tasso = pd.DataFrame()
            for key, c_group in group_c:
                try:
                    p_group = group_p.get_group(key)
                    record = c_group.copy()
                    record['Tipo di indicatore demografico'] = "Tasso di migratorietà verso l'estero"
                    record['Value'] = c_group['Value'].iloc[0] / \
                        p_group['Value'].iloc[0]
                    db_cancellati_tasso = pd.concat(
                        [db_cancellati_tasso, record])
                except:
                    pass
           # st.session_state['dataframes']['db_cancellati_tasso'] = db_cancellati_tasso
            st.write(
                "Ecco un estratto del database dei tassi di migratorietà verso l'estero")
            st.write(db_cancellati_tasso.head(5))

            # %% previsioni cancellazioni verso l'estero

            # scelta dei parametri
            st.write("Specifica i parametri del modello di previsione")
            col1, col2, col3 = st.columns(3)
            with col1:
                alpha = st.number_input("Coefficiente di smorzamento $\\alpha$",
                                        min_value=0.0, max_value=1.0, value=0.3, step=0.01)

            with col2:
                r = st.number_input("Coefficiente di convergenza all'equilibrio $r$",
                                    min_value=0.0, max_value=1.0, value=0.1, step=0.01)

            with col3:
                P = int(st.number_input("Periodi di previsione $P$",
                                        min_value=10, max_value=30, value=20, step=1))

            with col1:
                delta = st.number_input("Ampiezza dello scenario previsionale $\\delta$",
                                        min_value=-1.0, max_value=1.0, value=0.1, step=0.01)
            scenario = [+0.0, delta, -delta]  # parametri di scenario
            # %%
           # db_cancellati_tasso = st.session_state['dataframes']['db_cancellati_tasso']
            db_cancellati_tasso.rename(
                columns={"Value": "Mediano"}, inplace=True)
            db_cancellati_tasso["Alto"] = db_cancellati_tasso["Mediano"]
            db_cancellati_tasso["Basso"] = db_cancellati_tasso["Mediano"]
            db_group = db_cancellati_tasso.groupby('Unicode')
            # i codici univoci
            gp_code = db_cancellati_tasso['Unicode'].unique()
            if st.button(
                    'Voglio testare il modello previsionale per i cancellati su una serie casuale estratta'):
                k = random.choice(gp_code)
                grafico = "Sì"
                for i, serie in enumerate(["Mediano", "Alto", "Basso"]):
                    try:
                        plt = df_forecast(db_group, k, alpha, r, P,
                                          grafico, scenario[i], serie)
                        if i == 2:
                            st.pyplot(plt)
                    except:
                        pass

            if st.button("Procedi con l'aggiornamento degli scenari per i cancellati verso l'estero"):
                if "db_iscritti_forecast" in st.session_state['dataframes']:
                    del st.session_state['dataframes']['db_iscritti_forecast']
                grafico = "No"
                db_cancellati_tasso_forecast = pd.DataFrame()
                progress_bar = st.progress(0)
                tprogress = 0
                nprogress = len(gp_code)
                for k in gp_code:
                    for i, serie in enumerate(["Mediano", "Alto", "Basso"]):
                        try:
                            Yf = df_forecast(db_group, k, alpha, r, P,
                                             grafico, scenario[i], serie)
                            if serie == "Mediano":
                                db_cancellati_tasso_forecast = Yf
                            else:
                                db_cancellati_tasso_forecast[serie] = Yf[serie]
                        except:
                            pass
                    db_cancellati_tasso = pd.concat(
                        [db_cancellati_tasso, db_cancellati_tasso_forecast])
                    tprogress += 1
                    progress_bar.progress(tprogress/nprogress)
                # %%
                st.success("Elaborazione completata!")
                db_cancellati_tasso = db_cancellati_tasso.sort_values(
                    by=['Unicode', 'TIME'])  # ordinamento
                st.session_state['dataframes']['db_cancellati_tasso_forecast'] = db_cancellati_tasso
                st.write(
                    "Ecco un estratto del database previsinale per le cancellazioni verso l'estero")
                st.write(db_cancellati_tasso.head(10))
        # %% Modulo per gli iscritti verso l'estero
        if "db_cancellati_tasso_forecast" in st.session_state["dataframes"] and "db_iscritti_forecast" not in st.session_state["dataframes"]:
            st.error(
                "Manca lo scenario per le iscrizioni dall'estero")
            Input_file_4 = st.file_uploader(
                'Carica il file degli iscritti ISTAT_Migrazioni_1.csv, verifica che non vi siano cambiamenti nel nome dei campi e nel separatore')
            if Input_file_4:
                try:
                    db_iscritti = pd.read_csv(
                        Input_file_4, sep=',', encoding='utf-8')
                    remove = ['TIPO_DATO15', 'Tipo di trasferimento', 'Paese di cittadinanza', 'SEXISTAT1',
                              'Età', 'Seleziona periodo', 'Flag Codes', 'Flags', 'Territorio di origine', 'ITTER107_A']
                    db_iscritti.drop(columns=remove, inplace=True)
                    # Creazione codice univoco 'Unicode' e raggruppamento dei dati provinciali
                    Destinazione = ['FREIGN']
                    Nazionalità = ['TOTAL']
                    Sesso = ['totale']
                    Età = ['TOTAL']
                    db_iscritti = db_iscritti[db_iscritti['PROV_DEST_Z'].isin(
                        Destinazione)]  # solo origine Estero
                    db_iscritti = db_iscritti[~db_iscritti['ISO'].isin(
                        Nazionalità)]  # esclude il totale
                    db_iscritti = db_iscritti[~db_iscritti['Sesso'].isin(
                        Sesso)]  # esclude il totale
                    db_iscritti = db_iscritti[~db_iscritti['ETA1'].isin(
                        Età)]  # esclude il totale
                    # % definizione indici

                    db_iscritti['Unicode'] = db_iscritti['ITTER107_B'] + \
                        db_iscritti['ISO']+db_iscritti['Sesso'] + \
                        db_iscritti['ETA1']
                    st.write(
                        'Ecco un estratto del file sulle iscritizioni che hai caricato')
                    st.write(db_iscritti.head(5))
                except:
                    st.error('File errato o formato file non supportato')
                # % previsioni iscritti dall'estero
                db_iscritti.rename(columns={"Value": "Mediano"}, inplace=True)
                db_iscritti["Alto"] = db_iscritti["Mediano"]
                db_iscritti["Basso"] = db_iscritti["Mediano"]
                # %
                db_group = db_iscritti.groupby('Unicode')
                gp_code = db_iscritti['Unicode'].unique()  # i codici univoci
                # %
                if st.button(
                        "Voglio testare il modello previsionale per gli iscritti dall'estero su una serie casuale estratta"):
                    k = random.choice(gp_code)
                    grafico = "Sì"
                    for i, serie in enumerate(["Mediano", "Alto", "Basso"]):
                        try:
                            plt = df_forecast(db_group, k, alpha, r, P,
                                              grafico, scenario[i], serie)
                            if i == 2:
                                st.pyplot(plt)
                        except:
                            pass
                if st.button("Procedi con l'aggiornamento degli scenari per per gli iscritti dall'estero"):
                    grafico = "No"
                    db_iscritti_forecast = pd.DataFrame()
                    progress_bar = st.progress(0)
                    tprogress = 0
                    nprogress = len(gp_code)
                    for k in gp_code:
                        for i, serie in enumerate(["Mediano", "Alto", "Basso"]):
                            try:
                                Yf = df_forecast(db_group, k, alpha, r, P,
                                                 grafico, scenario[i], serie)
                                if serie == "Mediano":
                                    db_iscritti_forecast = Yf
                                else:
                                    db_iscritti_forecast[serie] = Yf[serie]
                            except:
                                pass
                        db_iscritti = pd.concat(
                            [db_iscritti, db_iscritti_forecast])
                        tprogress += 1
                        progress_bar.progress(tprogress/nprogress)
                # %
                    st.success("Elaborazione completata!")
                    db_iscritti = db_iscritti.sort_values(
                        by=['Unicode', 'TIME'])  # ordinamento
                    st.session_state['dataframes']['db_iscritti_forecast'] = db_iscritti
                    st.write(
                        "Ecco un estratto del database previsionale per gli iscritti dall'estero")
                    st.write(db_iscritti.head(10))
    # %% movimenti interni
    elif scelta == "Movimenti interni":
        Input_file_interm = st.file_uploader(
            'Carica il file dei cancellati ISTAT_Migrazioni_2.csv, verifica che non vi siano cambiamenti nel nome dei campi e nel separatore')
        if Input_file_interm:
            try:
                db_cancellati = pd.read_csv(
                    Input_file_interm, sep=',', encoding='utf-8')
                remove = ['TIPO_DATO15', 'Tipo di trasferimento', 'Paese di cittadinanza', 'SEXISTAT1', 'Età',
                          'PAESI_B', 'Seleziona periodo', 'Flag Codes', 'Flags', 'Stato estero di  destinazione']
                db_cancellati.drop(columns=remove, inplace=True)
                # Creazione codice univoco 'Unicode' e raggruppamento dei dati provinciali
                Destinazione = ['DRS', 'DP_SR']
                Nazionalità = ['TOTAL']
                Sesso = ['totale']
                Età = ['TOTAL']
                db_cancellati = db_cancellati[db_cancellati['PROV_DEST_Z'].isin(
                    Destinazione)]  # solo destinazione Estero
                db_cancellati = db_cancellati[~db_cancellati['ISO'].isin(
                    Nazionalità)]  # esclude il totale
                db_cancellati = db_cancellati[~db_cancellati['Sesso'].isin(
                    Sesso)]  # esclude il totale
                db_cancellati = db_cancellati[~db_cancellati['ETA1'].isin(
                    Età)]  # esclude il totale
                db_cancellati['Unicode'] = db_cancellati['ITTER107_A'] + \
                    db_cancellati['ISO'] + \
                    db_cancellati['Sesso']+db_cancellati['ETA1']
                # %% bisogna aggregare rispetto a 'PROV_DEST_Z'
                group_cols = [col for col in db_cancellati.columns if col not in [
                    'PROV_DEST_Z', 'Value']]
                tot_canc = db_cancellati.groupby(
                    group_cols)['Value'].sum().reset_index()
                tot_canc['PROV_DEST_Z'] = 'TOT'
                db_cancellati = tot_canc
                db_cancellati = db_cancellati.sort_values(
                    by=['Unicode', 'TIME'])  # ordinamento
                del tot_canc
                st.write(
                    "Ecco un estratto del file che hai caricato dopo l'elaborazione")
                st.write(db_cancellati.head(5))
            except:
                st.error('File errato o formato file non supportato')
                # %

            # %% definizione dei tassi di migratorietà
            group_c = db_cancellati.groupby(['Unicode', 'TIME'])
            db_pop_classe_final = st.session_state['dataframes']['db_pop_classe_final']
            group_p = db_pop_classe_final.groupby(['Unicode', 'TIME'])
            # %
            db_cancellati_interni_tasso = pd.DataFrame()
            for key, c_group in group_c:
                try:
                    p_group = group_p.get_group(key)
                    record = c_group.copy()
                    record['Tipo di indicatore demografico'] = "Tasso di migratorietà verso altre province"
                    record['Value'] = c_group['Value'].iloc[0] / \
                        p_group['Value'].iloc[0]
                    db_cancellati_interni_tasso = pd.concat(
                        [db_cancellati_interni_tasso, record])
                except:
                    pass
            # st.session_state['dataframes']['db_cancellati_interni_tasso'] = db_cancellati_interni_tasso
            st.write(
                "Ecco un estratto del database dei tassi di migratorietà interni")
            st.write(db_cancellati_interni_tasso.head(5))

            # %% previsioni cancellazioni verso comuni italiani al di fuori della provincia

            # scelta dei parametri
            st.write("Specifica i parametri del modello di previsione")
            col1, col2, col3 = st.columns(3)
            with col1:
                alpha = st.number_input("Coefficiente di smorzamento $\\alpha$",
                                        min_value=0.0, max_value=1.0, value=0.3, step=0.01)

            with col2:
                r = st.number_input("Coefficiente di convergenza all'equilibrio $r$",
                                    min_value=0.0, max_value=1.0, value=0.1, step=0.01)

            with col3:
                P = int(st.number_input("Periodi di previsione $P$",
                                        min_value=10, max_value=30, value=20, step=1))

            with col1:
                delta = st.number_input("Ampiezza dello scenario previsionale $\\delta$",
                                        min_value=-1.0, max_value=1.0, value=0.1, step=0.01)
            scenario = [+0.0, delta, -delta]  # parametri di scenario
            # %%
            # db_cancellati_interni_tasso = st.session_state['dataframes']['db_cancellati_interni_tasso']
            db_cancellati_interni_tasso.rename(
                columns={"Value": "Mediano"}, inplace=True)
            db_cancellati_interni_tasso["Alto"] = db_cancellati_interni_tasso["Mediano"]
            db_cancellati_interni_tasso["Basso"] = db_cancellati_interni_tasso["Mediano"]
            db_group = db_cancellati_interni_tasso.groupby('Unicode')
            # i codici univoci
            gp_code = db_cancellati_interni_tasso['Unicode'].unique()
            if st.button(
                    'Voglio testare il modello previsionale per i cancellati verso altre province su una serie casuale estratta'):
                k = random.choice(gp_code)
                grafico = "Sì"
                for i, serie in enumerate(["Mediano", "Alto", "Basso"]):
                    try:
                        plt = df_forecast(db_group, k, alpha, r, P,
                                          grafico, scenario[i], serie)
                        if i == 2:
                            st.pyplot(plt)
                    except:
                        pass

            if st.button("Procedi con l'aggiornamento degli scenari per i cancellati verso altre province"):
                grafico = "No"
                db_cancellati_tasso_forecast = pd.DataFrame()
                progress_bar = st.progress(0)
                tprogress = 0
                nprogress = len(gp_code)
                for k in gp_code:
                    for i, serie in enumerate(["Mediano", "Alto", "Basso"]):
                        try:
                            Yf = df_forecast(db_group, k, alpha, r, P,
                                             grafico, scenario[i], serie)
                            if serie == "Mediano":
                                db_cancellati_tasso_forecast = Yf
                            else:
                                db_cancellati_tasso_forecast[serie] = Yf[serie]
                        except:
                            pass
                    db_cancellati_interni_tasso = pd.concat(
                        [db_cancellati_interni_tasso, db_cancellati_tasso_forecast])
                    tprogress += 1
                    progress_bar.progress(tprogress/nprogress)
                # %%
                st.success("Elaborazione completata!")
                db_cancellati_interni_tasso = db_cancellati_interni_tasso.sort_values(
                    by=['Unicode', 'TIME'])  # ordinamento
                st.session_state['dataframes']['db_cancellati_interni_tasso_forecast'] = db_cancellati_interni_tasso
                st.write(
                    "Ecco un estratto del database previsinale per le cancellazioni verso l'estero")
                st.write(db_cancellati_interni_tasso.head(10))
        if "db_cancellati_interni_tasso_forecast" in st.session_state["dataframes"] and "db_Or_dest_dist" not in st.session_state["dataframes"]:
            st.error(
                "Per concludere, calcoliamo la matrice di distribuzione dei trasferiemnti di residenza verso altre province")
            Input_file_mov = st.file_uploader(
                'Carica il file dei cancellati ISTAT_Migrazioni_3.csv, verifica che non vi siano cambiamenti nel nome dei campi e nel separatore')
            if Input_file_mov:
                db_orig_dest = pd.read_csv(
                    Input_file_mov, sep=',', encoding='utf-8')
                remove = ['TIPO_DATO15', 'Paese di cittadinanza',
                          'SEXISTAT1', 'Seleziona periodo', 'Flag Codes', 'Flags']
                db_orig_dest.drop(columns=remove, inplace=True)
                # % Cancellati e iscritti dall'estero, italiani e stranieri
                Nazionalità = ['TOTAL']
                Sesso = ['totale']
                db_orig_dest = db_orig_dest[~db_orig_dest['ISO'].isin(
                    Nazionalità)]  # esclude il totale
                db_orig_dest = db_orig_dest[~db_orig_dest['Sesso'].isin(
                    Sesso)]  # esclude il totale
                # % elimino i trasferimenti di residenza all'interno della stessa provincia
                db_orig_dest = db_orig_dest[db_orig_dest['ITTER107_A']
                                            != db_orig_dest['ITTER107_B']]
                db_orig_dest.reset_index(drop=True, inplace=True)
                # %% bisogna calcolare la distribuzioen media origine_destinazione
                group_cols = [col for col in db_orig_dest.columns if col not in [
                    'TIME', 'Value']]
                tot_or_dest = db_orig_dest.groupby(
                    group_cols)['Value'].sum().reset_index()
                # %
                group_cols = [col for col in tot_or_dest.columns if col not in [
                    'ITTER107_B', 'Territorio di di destinazione', 'Value']]
                tot_or = tot_or_dest.groupby(
                    group_cols)['Value'].sum().reset_index()
                tot_or['Territorio di di destinazione'] = 'TOT'
                tot_or['ITTER107_B'] = 'TOT'
                # %
                tot_or_dest = pd.concat([tot_or_dest, tot_or])
                # %
                tot_or_dest = tot_or_dest.sort_values(
                    by=['ITTER107_A', 'ISO', 'Sesso', 'ITTER107_B'])
                tot_or_dest['Unicode'] = tot_or_dest['ITTER107_A'] + \
                    tot_or_dest['ISO']+tot_or_dest['Sesso']
                tot_or_dest = tot_or_dest[tot_or_dest['ITTER107_A']
                                          != tot_or_dest['ITTER107_B']]
                # %% calcolo della distribuzione media
                tot_or_dest['Quota'] = 0.0
                i = 1
                group_t = tot_or_dest.groupby('Unicode')
                db_tot_or_dest = pd.DataFrame()
                for k, t_group in group_t:
                    t_group = t_group.copy()
                    total = t_group.loc[t_group['ITTER107_B']
                                        == 'TOT', 'Value'].values[0]
                    t_group['Quota'] = t_group['Value'] / total
                    db_tot_or_dest = pd.concat([db_tot_or_dest, t_group])

                st.success("Elaborazione compeltata")
                db_tot_or_dest = db_tot_or_dest.sort_values(
                    by=['Unicode', 'ITTER107_B'])  # ordinamento
                st.session_state['dataframes']['db_Or_dest_dist'] = db_tot_or_dest
                st.write(
                    "Ecco un estratto del database origine-destinazione per i cambi di residenza")
                st.write(db_tot_or_dest.head(10))
