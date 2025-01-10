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


def df_forecast(db_group, k, scelta, alpha, r, P, grafico):
    Y = db_group.get_group(k)
    x = np.array(Y['Value'])

    # step 1: Applicare lo smoothing esponenziale per calcolare il valore di equilibrio
    if np.var(x) > 0:
        model_ses = sm.tsa.SimpleExpSmoothing(x)
        fitted_ses = model_ses.fit(
            smoothing_level=alpha, optimized=False)
        fitted_ses = fitted_ses.fittedvalues
    else:
        fitted_ses = [np.mean(x)] * len(x)
        equilibrium_value = np.mean(x)

    # step 2: Modello di Holt con trend lineare per generare previsioni
    if np.var(x) > 0:
        model_holt = sm.tsa.ExponentialSmoothing(
            x, trend="add", seasonal=None)
        fitted_holt = model_holt.fit()
        forecast_holt = fitted_holt.forecast(steps=P)
    else:
        forecast_holt = [np.mean(x)] * P

    if scelta == 'Passato':
        equilibrium_value = fitted_ses[-1]
    else:
        equilibrium_value = np.mean(forecast_holt[:10])
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
        plt.legend()
        plt.title(
            "Forecast con convergenza all'equilibrio serie " + k, fontsize=8)
        st.pyplot(plt)

    num_records = len(xf)
    # Estrai i parametri dimensionali
    dimensional_params = Y.iloc[0].drop(
        ['Value', 'TIME']).to_dict()
    new_data = {key: [value] * num_records for key,
                value in dimensional_params.items()}  # Replica i parametri
    new_data['Value'] = xf  # Popola il campo "Value" con xf
    new_data['TIME'] = xt    # Popola il campo "Time" con xt

    Yf = pd.DataFrame(new_data)
    # step 5: accoda al dataframe originale
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

st.title("Modulo 1: Aggiornamento matrici di fecondità e mortalità")
scelta = st.radio(
    "Scegli un'opzione:",
    options=["Matrice di fecondità", "Tassi di mortalità"]
)
# %% load file
if scelta == "Matrice di fecondità":
    Input_file = st.file_uploader(
        'Carica il file ISTAT_fecondità.csv, verifica che non vi siano cambiamenti nel nome dei campi e nel separatore')
    if Input_file:
        try:
            db_fecondità = pd.read_csv(Input_file, sep='|', encoding='utf-8')
            remove = ['Cittadinanza', 'Seleziona periodo', 'Età della madre',
                      'Flag Codes', 'Flags', 'Tipo dato', 'TIPO_DATO15']
            db_fecondità.drop(columns=remove, inplace=True)
            # Creazione codice univoco 'Unicode' e raggruppamento dei dati provinciali
            db_fecondità['Unicode'] = db_fecondità['ITTER107'] + \
                db_fecondità['CITTADINANZA']+db_fecondità['ETA1']
            db_group = db_fecondità.groupby('Unicode')
            gp_code = db_fecondità['Unicode'].unique()  # i codici univoci
            gp_code = sorted(gp_code)
            db_fecondità = db_fecondità.sort_values(
                by=['Unicode', 'TIME'])  # ordinamento
            st.write('Ecco un estratto del file che hai caricato')
            st.write(db_fecondità.head(5))
            test_file = 1
        except:
            st.error('Formato file non supportato')
            test_file = 0
            # %
        if test_file == 1:
            st.write("Specifica i parametri del modello di smoothing")
            col1, col2, col3 = st.columns(3)
            with col1:
                alpha = st.number_input("Coefficiente di smorzamento $\\alpha$",
                                        min_value=0.0, max_value=1.0, value=0.3, step=0.01)
                # scelta dell'equilibrio
                scelta = st.selectbox(
                    "Scegli un modello di equilibrio:",
                    ["Passato", "Futuro"])
            with col2:
                r = st.number_input("Coefficiente di convergenza all'equilibrio $r$",
                                    min_value=0.0, max_value=1.0, value=0.3, step=0.01)

            with col3:
                P = int(st.number_input("Periodi di previsione $P$",
                                        min_value=10, max_value=30, value=20, step=1))

            if st.button(
                    'Voglio testare il modello su una serie casuale estratta dal database'):
                grafico = "Sì"
                k = random.choice(gp_code)
                df_forecast(db_group, k, scelta, alpha, r, P, grafico)

            if st.button("Procedi con l'aggiornamento delle previsioni"):
                grafico = "No"
                progress_bar = st.progress(0)
                tprogress = 0
                nprogress = len(gp_code)
                for k in gp_code[:20]:
                    Yf = df_forecast(db_group, k, scelta, alpha, r, P, grafico)
                    # step 5: accoda al dataframe originale
                    db_fecondità = pd.concat(
                        [db_fecondità, Yf], ignore_index=True)
                    tprogress += 1
                    progress_bar.progress(tprogress/nprogress)
                st.success("Elaborazione completata!")
                db_fecondità = db_fecondità.sort_values(
                    by=['Unicode', 'TIME'])  # riordinamento
                st.write(
                    'Ecco un estratto del database aggiornato con le previsioni')
                st.write(db_fecondità.head(P*2))
                st.session_state['dataframes']['db_fecondità_forecast'] = db_fecondità
                st.write("File salvati nella sessione")
else:
    # %
    Input_file_2 = st.file_uploader(
        'Carica il file ISTAT_mortalità_pm.csv, verifica che non vi siano cambiamenti nel nome dei campi e nel separatore')
    if Input_file_2:
        try:
            db_mortalità = pd.read_csv(
                Input_file_2, sep='|', encoding='utf-8', decimal=',')
            remove = ['Seleziona periodo', 'Età e classi di età', 'SEXISTAT1', 'Funzioni biometriche',
                      'Flag Codes', 'Flags', 'TIPO_DATO15']
            db_mortalità.drop(columns=remove, inplace=True)
            # Creazione codice univoco 'Unicode' e raggruppamento dei dati provinciali
            db_mortalità['Unicode'] = db_mortalità['ITTER107'] + \
                db_mortalità['Sesso']+db_mortalità['ETA1']
            db_group = db_mortalità.groupby('Unicode')
            gp_code = db_mortalità['Unicode'].unique()  # i codici univoci
            db_mortalità = db_mortalità.sort_values(
                by=['Unicode', 'TIME'])  # ordinamento
            st.write('Ecco un estratto del file che hai caricato')
            st.write(db_mortalità.head(5))
            test_file_2 = 1
        except:
            st.error('Formato file non supportato')
            test_file_2 = 0
        Input_file_3 = st.file_uploader(
            'Carica il file ISTAT_mortalità_sop.csv, verifica che non vi siano cambiamenti nel nome dei campi e nel separatore')

        if Input_file_3:
            try:
                db_sopravviventi = pd.read_csv(
                    Input_file_3, sep='|', encoding='utf-8', decimal=',')
                remove = ['Seleziona periodo', 'Età e classi di età', 'SEXISTAT1', 'Funzioni biometriche',
                          'Flag Codes', 'Flags', 'TIPO_DATO15']
                db_sopravviventi.drop(columns=remove, inplace=True)
                # Creazione codice univoco 'Unicode' e raggruppamento dei dati provinciali
                db_sopravviventi['Unicode'] = db_sopravviventi['ITTER107'] + \
                    db_sopravviventi['Sesso']+db_sopravviventi['ETA1']
                db_sopravviventi = db_sopravviventi.sort_values(
                    by=['Unicode', 'TIME'])  # ordinamento
                st.write('Ecco un estratto del file che hai caricato')
                st.write(db_sopravviventi.head(5))
                test_file_3 = 1
            except:
                st.error('Formato file non supportato')
                test_file_3 = 0
            # calcolo della probabilità di morte maggiore di 100
            if "db_mortalità_100" not in st.session_state["dataframes"]:
                if st.button("Procedi al calcolo della probabilità di morte per 100 anni e più"):
                    # per ogni territorio, per ogni sesso, per ogni anno, selezionare le età superiori o uguali a 100 anni per mortlità e sopravviventi
                    group_m = db_mortalità.groupby(
                        ['ITTER107', 'Territorio', 'Sesso', 'TIME'])
                    group_s = db_sopravviventi.groupby(
                        ['ITTER107', 'Territorio', 'Sesso', 'TIME'])
                    età_interesse = ['Y100', 'Y101', 'Y102', 'Y103', 'Y104', 'Y105', 'Y106', 'Y107', 'Y108',
                                     'Y109', 'Y110', 'Y111', 'Y112', 'Y113', 'Y114', 'Y115', 'Y116', 'Y117', 'Y118', 'Y119']
                    db_mortalità_ge100 = pd.DataFrame(
                        columns=['ITTER107', 'Territorio', 'Sesso', 'ETA1', 'TIME',  'Value', 'Unicode'])
                    for i, (key, current_group_m) in enumerate(group_m):
                        try:
                            current_group_s = group_s.get_group(key)
                            pm_100 = current_group_m.loc[current_group_m['ETA1'].isin(
                                età_interesse), 'Value'].to_numpy()
                            s_100 = current_group_s.loc[current_group_s['ETA1'].isin(
                                età_interesse), 'Value'].to_numpy()
                            try:
                                prob_ge100 = np.sum(
                                    pm_100 * s_100)/np.sum(s_100)
                            except ZeroDivisionError:
                                prob_ge100 = 0
                        except KeyError:
                            prob_ge100 = 0
                        record = {
                            'ITTER107': key[0],
                            'Territorio': key[1],
                            'Sesso': key[2],
                            'ETA1': 'Yge100',     # Imposta ETA1 a "ge100"
                            'TIME': key[3],
                            'Value': prob_ge100,  # Inserisci il valore calcolato
                            'Unicode': key[0]+key[2]+'Yge100'
                        }
                        db_prob_ge100 = pd.DataFrame([record])
                        if db_mortalità_ge100.empty:
                            db_mortalità_ge100 = db_prob_ge100
                        else:
                            db_mortalità_ge100 = pd.concat(
                                [db_mortalità_ge100, db_prob_ge100], ignore_index=True)
                    # % Final task
                    # accoda il risultato ottenuto
                    db_mortalità_100 = pd.concat(
                        [db_mortalità, db_mortalità_ge100], ignore_index=True)
                    db_mortalità_100 = db_mortalità_100.sort_values(
                        by=['Unicode', 'TIME'])  # ordinamento
                    db_mortalità_100 = db_mortalità_100[~db_mortalità_100['ETA1'].isin(
                        età_interesse)]  # esclude le età superiori a 100
                    st.success("Calcolo effettuato")
                    st.session_state['dataframes']['db_mortalità_100'] = db_mortalità_100
                    st.write("File salvati in sessione")

            # Previsione
            if "db_mortalità_100" in st.session_state["dataframes"]:
                st.write(
                    "Specifica i parametri del modello di smoothing per la previsione della probabilità di morte")
                col4, col5, col6 = st.columns(3)
                with col4:
                    alphaM = st.number_input("Coefficiente di smorzamento  $\\alpha_M$",
                                             min_value=0.0, max_value=1.0, value=0.3, step=0.01)
                    # scelta dell'equilibrio
                    scelta = st.selectbox(
                        "Scegli un modello di equilibrio :",
                        ["Passato", "Futuro"])
                with col5:
                    rM = st.number_input("Coefficiente di convergenza all'equilibrio $r_M$",
                                         min_value=0.0, max_value=1.0, value=0.3, step=0.01)

                with col6:
                    PM = int(st.number_input("Periodi di previsione $P_M$",
                                             min_value=10, max_value=30, value=20, step=1))
                if st.button(
                        'Voglio testare il modello su una serie casuale estratta'):
                    db_mortalità_100 = st.session_state['dataframes']['db_mortalità_100']
                    db_group = db_mortalità_100.groupby('Unicode')
                    # i codici univoci
                    gp_code = db_mortalità_100['Unicode'].unique()
                    k = random.choice(gp_code)
                    grafico = "Sì"

                    df_forecast(db_group, k, scelta, alphaM, rM, PM, grafico)

                if st.button("Procedi con l'aggiornamento delle previsioni per la mortalità"):
                    grafico = "No"
                    db_mortalità_100 = st.session_state['dataframes']['db_mortalità_100']
                    db_group = db_mortalità_100.groupby('Unicode')
                    # i codici univoci
                    gp_code = db_mortalità_100['Unicode'].unique()
                    progress_bar = st.progress(0)
                    tprogress = 0
                    nprogress = len(gp_code)
                    for k in gp_code[0:20]:
                        Yf = df_forecast(db_group, k, scelta,
                                         alphaM, rM, PM, grafico)
                        # accoda al dataframe  originale
                        db_mortalità_100_forecast = pd.concat(
                            [db_mortalità_100, Yf], ignore_index=True)
                        tprogress += 1
                        progress_bar.progress(tprogress/nprogress)
                    st.success("Elaborazione completata!")
                    db_mortalità_100_forecast = db_mortalità_100_forecast.sort_values(
                        by=['Unicode', 'TIME'])  # riordinamento
                    st.write(
                        'Ecco un estratto del database aggiornato con le previsioni')
                    st.write(db_mortalità_100_forecast.head(PM*2))

                    st.session_state['dataframes']['db_mortalità_100_forecast'] = db_mortalità_100_forecast
                    st.write("File salvati nella sessione")
