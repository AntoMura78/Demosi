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
# %%
st.header(':blue[DemoSi updater]', divider='blue')
st.title("Modulo 1: aggiornamento matrice di fencondità")
# %% load file
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
        # %%
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

            k = random.choice(gp_code)
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

        if st.button("Procedi con l'aggiornamento delle previsioni"):
            progress_bar = st.progress(0)
            tprogress = 0
            nprogress = len(gp_code)
            for k in gp_code:
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
                db_fecondità = pd.concat([db_fecondità, Yf], ignore_index=True)
                tprogress += 1
                progress_bar.progress(tprogress/nprogress)
            st.success("Elaborazione completata!")
            db_fecondità = db_fecondità.sort_values(
                by=['Unicode', 'TIME'])  # riordinamento
            st.write('Ecco un estratto del database aggiornato con le previsioni')
            st.write(db_fecondità.head(100))
            dbf_csv = db_fecondità.to_csv(
                sep="|", index=False)
            st.download_button('Scarica il file in formato CSV',
                               data=dbf_csv, file_name='ISTAT_fecondità_forecast.csv')
