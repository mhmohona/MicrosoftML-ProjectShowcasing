# Store-Item-Demand-Forecasting-Challenge using Facebook Prophet--by Priyanka Gupta
Exploratory data Analysis along with Time Series Forecasting

Link of Kaggle Submission: https://www.kaggle.com/pg1007/time-series-analysis-prophet

Use above link for better visualisations

Time Series Analysis is most important tool to predict future sales. As most of the time retail shop undergo the process of understocking and overstocking, to overcome this challenge i predicted the sales of 10 different items at 50 different stores.This helps the business to provide the best customer experience and avoid getting into losses, thus ensuring the store is sustainable for operation.

Firstly performed--> Explorative Data Analysis and find out how sales is dependent on item and  store.
Some store have high sales as compared to other whereas some item were popular on different stores.

Some of the observations that can be viewed are:



*Image of Average sales of items*
![Alt Text](https://www.kaggleusercontent.com/kf/43069285/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Ost32AuAcbxz4X3AzQVAVQ.vfQdt20SbUTAsaA0EjhKcigFLxpC0KQ_ICZVMHM3cy18y1qUmaTv3qDmR5JB2_g91Li257iYGmRKnyUrQGk6qSXAurbMFM_sw6yBuxh96vf9t94eY2P9r0xvtWxHcqjbaHJnN8S9nWdVqa69VbOJJJX6fkr9uDaSy6eVb27BvGyKv7LSZjjVhiqLozwQ6FAkvFv3Jn-NbqLb0U9kmabGq3CufrFMmu2_wmUJNjjgVDplg857irbOf2DM4Yabnv9kvLV_rsFr-AAtSf25i_yD7yeSoNPKUOXe1AsEErI56HTLdMRkhS3o8JKRISQtKk9Bavfjt5VCigCAEs8N81hMVO4XZnXb9HsOfj7iLkE22DLKXk2LZhnuPTGd-JFwjW0C-Zr_b21gp3Y3nIB16j9TtkNq_9kMFAEiQyjtwwmW0F6Zpf7iYAVZtDSsnDfVtCVUBrLZcPQuEmd2fCCRW6OnuMpniTG5XuPln75RBKbVRiUftENg7lt_iUEU-7j1a3kwWzV4SSsZ34eN6jauBkf0GBZLXUrpu7xE2PRx1vtK_b7YtIaMfVNnuUcTpBLe6g84WLL9eOVGgeDsgMiVTp7tKKlo1OHUHPAEDs7PhVUVJnSIAUCD9Q9GTwewo14Q6YkjWPWzka4OuwjPf3OGxxbmfQ.4Y_qN8Bopm2nFDCymwZsqQ/__results___files/__results___15_1.png)


*Image of average sales at each store* 

![Alt Text](https://www.kaggleusercontent.com/kf/43069285/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Ost32AuAcbxz4X3AzQVAVQ.vfQdt20SbUTAsaA0EjhKcigFLxpC0KQ_ICZVMHM3cy18y1qUmaTv3qDmR5JB2_g91Li257iYGmRKnyUrQGk6qSXAurbMFM_sw6yBuxh96vf9t94eY2P9r0xvtWxHcqjbaHJnN8S9nWdVqa69VbOJJJX6fkr9uDaSy6eVb27BvGyKv7LSZjjVhiqLozwQ6FAkvFv3Jn-NbqLb0U9kmabGq3CufrFMmu2_wmUJNjjgVDplg857irbOf2DM4Yabnv9kvLV_rsFr-AAtSf25i_yD7yeSoNPKUOXe1AsEErI56HTLdMRkhS3o8JKRISQtKk9Bavfjt5VCigCAEs8N81hMVO4XZnXb9HsOfj7iLkE22DLKXk2LZhnuPTGd-JFwjW0C-Zr_b21gp3Y3nIB16j9TtkNq_9kMFAEiQyjtwwmW0F6Zpf7iYAVZtDSsnDfVtCVUBrLZcPQuEmd2fCCRW6OnuMpniTG5XuPln75RBKbVRiUftENg7lt_iUEU-7j1a3kwWzV4SSsZ34eN6jauBkf0GBZLXUrpu7xE2PRx1vtK_b7YtIaMfVNnuUcTpBLe6g84WLL9eOVGgeDsgMiVTp7tKKlo1OHUHPAEDs7PhVUVJnSIAUCD9Q9GTwewo14Q6YkjWPWzka4OuwjPf3OGxxbmfQ.4Y_qN8Bopm2nFDCymwZsqQ/__results___files/__results___23_1.png)

**Time Series Analysis**

**Stationary Time Series**

A stationary time series is one whose properties do not depend on the time at which the series is observed.Thus, time series with trends, or with seasonality, are not stationary — the trend and seasonality will affect the value of the time series at different times. Stationary time plots will show the series to be roughly horizontal (although some cyclic behaviour is possible), with constant variance.


**The Prophet Forecasting Model**

We use a decomposable time series model with three main model components: trend, seasonality, and holidays. They are combined in the following equation:

                                            y(t)=g(t)+s(t)+h(t)+et

    g(t): piecewise linear or logistic growth curve for modelling non-periodic changes in time series
    s(t): periodic changes (e.g. weekly/yearly seasonality)
    h(t): effects of holidays (user provided) with irregular schedules
    εt: error term accounts for any unusual changes not accommodated by the model

Using time as a regressor, Prophet is trying to fit several linear and non linear functions of time as components. Modeling seasonality means defining seasonality. It is an additive component is the same approach taken by exponential smoothing in Holt-Winters technique . We are, in effect, framing the forecasting problem as a curve-fitting exercise rather than looking explicitly at the time based dependence of each observation within a time series.

**Seasonality**

A Seasonality time series is observed in given dataset.It refers to periodic fluctuations.Online sales increase during Holidays and festive days like Christmas before slowing down again.To fit and forecast the effects of seasonality, prophet relies on fourier series to provide a flexible model. Seasonal effects s(t) are approximated by the following function:

   ![Alt Text](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/05/ts_seasonal_Capture.jpg)

**Trend**

Trend is modelled by fitting a piece wise linear curve over the trend or the non-periodic part of the time series. The linear fitting exercise ensures that it is least affected by spikes/missing data.

**Holidays and events**

Holidays and events incur predictable shocks to a time series. For instance, Diwali in India occurs on a different day each year and a large portion of the population buy a lot of new items during this period.

Prophet allows the analyst to provide a custom list of  past and future events. A window around such days are considered separately and additional parameters are fitted to model the effect of holidays and events.

*Following Graph shows seasonality and predictions for next 90  days using prophet*
![Alt Text](https://www.kaggleusercontent.com/kf/43069285/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Ost32AuAcbxz4X3AzQVAVQ.vfQdt20SbUTAsaA0EjhKcigFLxpC0KQ_ICZVMHM3cy18y1qUmaTv3qDmR5JB2_g91Li257iYGmRKnyUrQGk6qSXAurbMFM_sw6yBuxh96vf9t94eY2P9r0xvtWxHcqjbaHJnN8S9nWdVqa69VbOJJJX6fkr9uDaSy6eVb27BvGyKv7LSZjjVhiqLozwQ6FAkvFv3Jn-NbqLb0U9kmabGq3CufrFMmu2_wmUJNjjgVDplg857irbOf2DM4Yabnv9kvLV_rsFr-AAtSf25i_yD7yeSoNPKUOXe1AsEErI56HTLdMRkhS3o8JKRISQtKk9Bavfjt5VCigCAEs8N81hMVO4XZnXb9HsOfj7iLkE22DLKXk2LZhnuPTGd-JFwjW0C-Zr_b21gp3Y3nIB16j9TtkNq_9kMFAEiQyjtwwmW0F6Zpf7iYAVZtDSsnDfVtCVUBrLZcPQuEmd2fCCRW6OnuMpniTG5XuPln75RBKbVRiUftENg7lt_iUEU-7j1a3kwWzV4SSsZ34eN6jauBkf0GBZLXUrpu7xE2PRx1vtK_b7YtIaMfVNnuUcTpBLe6g84WLL9eOVGgeDsgMiVTp7tKKlo1OHUHPAEDs7PhVUVJnSIAUCD9Q9GTwewo14Q6YkjWPWzka4OuwjPf3OGxxbmfQ.4Y_qN8Bopm2nFDCymwZsqQ/__results___files/__results___33_0.png)


