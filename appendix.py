

# fig,axs = plt.subplots(nrows=4,ncols=3,figsize = (14,20), tight_layout= True, sharey=True)
# axs = axs.flatten()
# axs[0].hist(dtfrm2['SPEED_UP_MAX'])
# axs[0].set_yticks([0,500,1000,1500,2000,2500, 3000, 3500, 4000],['0','500','1,000','1,500','2,000','2,500','3,000','3,500','4,000'])
# axs[0].set_ylabel('Orders')
# axs[0].set_xlabel('Maximum Speed up')
# axs[0].set_title('Histogram of Maximum Speed up values')

# axs[3].hist(dtfrm2['SPEED_UP_MEAN'])
# axs[3].set_yticks([0,500,1000,1500,2000,2500, 3000, 3500, 4000],['0','500','1,000','1,500','2,000','2,500','3,000','3,500','4,000'])
# axs[3].set_ylabel('Orders')
# axs[3].set_xlabel('Mean Speed up')
# axs[3].set_title('Histogram of Mean Speed up values')

# #
# axs[1].hist(dtfrm2['PRESSURE_UP_MAX'], color = 'c')
# axs[1].set_yticks([0,500,1000,1500,2000,2500, 3000, 3500, 4000],['0','500','1,000','1,500','2,000','2,500','3,000','3,500','4,000'])
# #axs[1].set_ylabel('Orders')
# axs[1].set_xlabel('Maximum Pressure up')
# axs[1].set_title('Histogram of Maximum Pressure up values')

# axs[4].hist(dtfrm2['PRESSURE_UP_MEAN'], color = 'c')
# axs[4].set_yticks([0,500,1000,1500,2000,2500, 3000, 3500, 4000],['0','500','1,000','1,500','2,000','2,500','3,000','3,500','4,000'])
# #axs[4].set_ylabel('Orders')
# axs[4].set_xlabel('Mean Pressure up')
# axs[4].set_title('Histogram of Mean Pressure up values')

# #

# axs[2].hist(dtfrm2['NOISE_UP_MAX'], color = 'deepskyblue')
# axs[2].set_yticks([0,500,1000,1500,2000,2500, 3000, 3500, 4000],['0','500','1,000','1,500','2,000','2,500','3,000','3,500','4,000'])
# #axs[2].set_ylabel('Orders')
# axs[2].set_xlabel('Maximum Noise up')
# axs[2].set_title('Histogram of Maximum Noise up values')

# axs[5].hist(dtfrm2['NOISE_UP_MEAN'],color = 'deepskyblue')
# axs[5].set_yticks([0,500,1000,1500,2000,2500, 3000, 3500, 4000],['0','500','1,000','1,500','2,000','2,500','3,000','3,500','4,000'])
# #axs[5].set_ylabel('Orders')
# axs[5].set_xlabel('Mean Noise up')
# axs[5].set_title('Histogram of Mean Noise up values')
# ###
# axs[6].hist(dtfrm2['SPEED_DOWN_MAX'], color = 'g')
# axs[6].set_yticks([0,500,1000,1500,2000,2500, 3000, 3500, 4000],['0','500','1,000','1,500','2,000','2,500','3,000','3,500','4,000'])
# axs[6].set_ylabel('Orders')
# axs[6].set_xlabel('Maximum Speed down')
# axs[6].set_title('Histogram of Maximum Speed down values')

# axs[9].hist(dtfrm2['SPEED_DOWN_MEAN'], color = 'g')
# axs[9].set_yticks([0,500,1000,1500,2000,2500, 3000, 3500, 4000],['0','500','1,000','1,500','2,000','2,500','3,000','3,500','4,000'])
# axs[9].set_ylabel('Orders')
# axs[9].set_xlabel('Mean Speed down')
# axs[9].set_title('Histogram of Mean Speed down values')

# #
# axs[7].hist(dtfrm2['PRESSURE_DOWN_MAX'], color = 'limegreen')
# axs[7].set_yticks([0,500,1000,1500,2000,2500, 3000, 3500, 4000],['0','500','1,000','1,500','2,000','2,500','3,000','3,500','4,000'])
# #axs[7].set_ylabel('Orders')
# axs[7].set_xlabel('Maximum Pressure down')
# axs[7].set_title('Histogram of Maximum Pressure down values')

# axs[10].hist(dtfrm2['PRESSURE_DOWN_MEAN'], color = 'limegreen')
# axs[10].set_yticks([0,500,1000,1500,2000,2500, 3000, 3500, 4000],['0','500','1,000','1,500','2,000','2,500','3,000','3,500','4,000'])
# #axs[9].set_ylabel('Orders')
# axs[10].set_xlabel('Mean Pressure down')
# axs[10].set_title('Histogram of Mean Pressure down values')

# #

# axs[8].hist(dtfrm2['NOISE_DOWN_MAX'], color = 'lime')
# axs[8].set_yticks([0,500,1000,1500,2000,2500, 3000, 3500, 4000],['0','500','1,000','1,500','2,000','2,500','3,000','3,500','4,000'])
# #axs[8].set_ylabel('Orders')
# axs[8].set_xlabel('Maximum Noise down')
# axs[8].set_title('Histogram of Maximum Noise down values')

# axs[11].hist(dtfrm2['NOISE_DOWN_MEAN'], color = 'lime')
# axs[11].set_yticks([0,500,1000,1500,2000,2500, 3000, 3500, 4000],['0','500','1,000','1,500','2,000','2,500','3,000','3,500','4,000'])
# #axs[11].set_ylabel('Orders')
# axs[11].set_xlabel('Mean Noise down')
# axs[11].set_title('Histogram of Mean Noise down values')

# plt.savefig('plots', dpi = 220)






# from matplotlib import gridspec

# kmeans_flow_rate = KMeans(len(flow_rate)).fit(dtfrm2[['SPEED_UP_MEAN','SPEED_UP_MAX','PRESSURE_UP_MEAN','PRESSURE_UP_MAX','NOISE_UP_MEAN','NOISE_UP_MAX']])
# kmeans_pumps = KMeans(len(hand_pump)).fit(dtfrm2[['SPEED_UP_MEAN','SPEED_UP_MAX','PRESSURE_UP_MEAN','PRESSURE_UP_MAX','NOISE_UP_MEAN','NOISE_UP_MAX']])
# kmeans_switch = KMeans(len(switch)).fit(dtfrm2[['SPEED_UP_MEAN','SPEED_UP_MAX','PRESSURE_UP_MEAN','PRESSURE_UP_MAX','NOISE_UP_MEAN','NOISE_UP_MAX']])
# kmeans_tanks = KMeans(len(tanks)).fit(dtfrm2[['SPEED_UP_MEAN','SPEED_UP_MAX','PRESSURE_UP_MEAN','PRESSURE_UP_MAX','NOISE_UP_MEAN','NOISE_UP_MAX']])
# kmeans_valves = KMeans(len(valves)).fit(dtfrm2[['SPEED_UP_MEAN','SPEED_UP_MAX','PRESSURE_UP_MEAN','PRESSURE_UP_MAX','NOISE_UP_MEAN','NOISE_UP_MAX']])


# ig = plt.figure(figsize=(17, 10)) 
# gs = gridspec.GridSpec(2, 3) 
# #ax = ax.ravel(order='C')
# ax1 = plt.subplot(gs[0,:2])
# ax2 = plt.subplot(gs[0,2])
# ax3 = plt.subplot(gs[1,0])
# ax4 = plt.subplot(gs[1,1])
# ax5 = plt.subplot(gs[1,2])


# ax1.scatter(kmeans_flow_rate.labels_, dtfrm2['FLOW_RATE'], c = 'g')
# ax1.set_title('Flow rate')
# ax1.set_xlabel('Clusters')
# ax1.set_ylabel('Flow rates to numerical')

# ax2.scatter(kmeans_pumps.labels_, dtfrm2['HAND_PUMP'])
# ax2.set_title('Hand pump')
# ax2.set_xlabel('Clusters')
# ax2.set_ylabel('Pumps to numerical')

# ax3.scatter(kmeans_switch.labels_, dtfrm2['PRESSURE_SWITCH'])
# ax3.set_title('Pressure Switch')
# ax3.set_xlabel('Clusters')
# ax3.set_ylabel('Switches to numerical')

# ax4.scatter(kmeans_tanks.labels_, dtfrm2['TANK_TYPE'])
# ax4.set_title('Tank Type')
# ax4.set_xlabel('Clusters')
# ax4.set_ylabel('Tanks to numerical')

# ax5.scatter(kmeans_valves.labels_, dtfrm2['BLOCK_VALVE'])
# ax5.set_title('Block valve')
# ax5.set_xlabel('Clusters')
# ax5.set_ylabel('Valves to numerical')

# plt.savefig('kmeans.png', dpi= 220)




####Mean and maximum

# fig,axs = plt.subplots(nrows=2,ncols=3,figsize = (8,11), tight_layout= True, sharey=True)
# axs = axs.flatten()
# axs[0].hist(pd.concat([dtfrm2['SPEED_UP_MAX'],dtfrm2['SPEED_DOWN_MAX']]), bins = 100)
# axs[0].set_yticks([0,200,400,600,800,1000,1200, 1400],['0','200','400','600','800','1,000','1,200','1,400'])
# axs[0].set_ylabel(r'$Orders (7200) \times Operations (2)$')
# axs[0].set_xlabel('Maximum Speed (m/s)')
# axs[0].set_title('Maximum Speed values')

# axs[3].hist(pd.concat([dtfrm2['SPEED_UP_MEAN'], dtfrm2['SPEED_DOWN_MEAN']]), bins = 100)
# axs[3].set_yticks([0,200,400,600,800,1000,1200, 1400],['0','200','400','600','800','1,000','1,200','1,400'])
# axs[3].set_ylabel(r'$Orders (7200) \times Operations (2)$')
# axs[3].set_xlabel('Mean Speed (m/s)')
# axs[3].set_title('Mean Speed values')

# #
# axs[2].hist(pd.concat([dtfrm2['PRESSURE_UP_MAX'], dtfrm2['PRESSURE_DOWN_MAX']]), color = 'c', bins = 100)
# axs[2].set_yticks([0,200,400,600,800,1000,1200, 1400],['0','200','400','600','800','1,000','1,200','1,400'])
# #axs[1].set_ylabel('Orders')
# axs[2].set_xlabel('Maximum Pressure (bar)')
# axs[2].set_title('Maximum Pressure values')

# axs[5].hist(pd.concat([dtfrm2['PRESSURE_UP_MEAN'],dtfrm2['PRESSURE_DOWN_MEAN']]), color = 'c', bins = 100)
# axs[5].set_yticks([0,200,400,600,800,1000,1200, 1400],['0','200','400','600','800','1,000','1,200','1,400'])
# #axs[4].set_ylabel('Orders')
# axs[5].set_xlabel('Mean Pressure (bar)')
# axs[5].set_title('Mean Pressure values')

# #

# axs[1].hist(pd.concat([dtfrm2['NOISE_UP_MAX'],dtfrm2['NOISE_DOWN_MAX']]), color = 'deepskyblue', bins = 100)
# axs[1].set_yticks([0,200,400,600,800,1000,1200, 1400],['0','200','400','600','800','1,000','1,200','1,400'])
# #axs[2].set_ylabel('Orders')
# axs[1].set_xlabel('Maximum Noise (dB)')
# axs[1].set_title('Maximum Noise values')

# axs[4].hist(pd.concat([dtfrm2['NOISE_UP_MEAN'], dtfrm2['NOISE_DOWN_MEAN']]),color = 'deepskyblue', bins = 100)
# axs[4].set_yticks([0,200,400,600,800,1000,1200, 1400],['0','200','400','600','800','1,000','1,200','1,400'])
# #axs[5].set_ylabel('Orders')
# axs[4].set_xlabel('Mean Noise (dB)')
# axs[4].set_title('Mean Noise values')
# ###


# plt.savefig('plots', dpi = 220)

######timesteps

# l = []
# for i in range(len(orders)):
#     l.append(len(dtfrm.loc[orders[i]]))
# plt.hist(l)
# plt.title('Time steps in each order')
# plt.xlabel('number of time steps')
# plt.xticks([200,400,600,800,1000], ['200','400','600','800','1,000'])
# plt.yticks([1000,2000,3000,4000,5000, 6000], ['1,000','2,000','3,000','4,000','5,000','6,000'])
# plt.ylabel('number of orders')
# plt.savefig('timesteps', dpi = 220)






#################anamalies



# fig, axes = plt.subplots(figsize=(15,10), nrows=2, ncols=3, tight_layout= True)

# dtfrm.replace(0,np.nan).loc[orders[(len(x)+9)//2]][['TIME','SPEED_UP']].plot(x = 'TIME',ax = axes[0,0])
# axes[0,0].set_title('Example 1')

# dtfrm.replace(0,np.nan).loc[orders[(len(x)+3)//2]][['TIME','SPEED_DOWN']].plot(x =  'TIME',ax = axes[0,1])
# axes[0,1].set_title('Example 2')

# dtfrm.replace(0,np.nan).loc[orders[(len(x)+24)//2]][['TIME','SPEED_DOWN']].plot(x =  'TIME',ax = axes[0,2])
# axes[0,2].set_title('Example 3')
# dtfrm.replace(0,np.nan).loc[211786][['TIME','PRESSURE_UP']].plot(x = 'TIME',ax = axes[1,0])
# axes[1,0].set_title('Example 4')

# dtfrm.replace(0,np.nan).loc[211832][['TIME','PRESSURE_UP']].plot(x =  'TIME',ax = axes[1,1])
# axes[1,1].set_title('Example 5')
# dtfrm.replace(0,np.nan).loc[211893][['TIME','PRESSURE_UP']].plot(x =  'TIME',ax = axes[1,2])
# axes[1,2].set_title('Example 6')

# plt.savefig('anomalies')





###########dummy data
# fig, axes = plt.subplots(figsize=(10,16), nrows=3, ncols=2, tight_layout= True, sharex=True)
# col = -1
# l = [672,1080] 
# l2 = ['m/s','dB','bar']
# for i in range(3):
#     col += 1
#     for k in range(2):
#         if col == 0:
#             n = 0
#             p = 1
#         elif col == 1:
#             n = 1
#             p = 91.2
#         else:
#             n = 2
#             p = 53.98
        
#         axes[i,k].plot(test[l[k],:,n]*p,label = 'Normal data', linewidth = 3)
#         axes[i,k].plot(dummys_array[l[k],:,n]*p,label = 'Synthesized anomalous data')
#         axes[i,k].legend(loc = 'lower right')
#         if i == 2:
#             axes[i,k].set_xlabel('time(s)')
#         if k == 0:
#             axes[i,k].set_ylabel(l2[i])



# plt.savefig('dummy_data')


#usefull index in dummy [1080, 672, 9, 24, 3, [211786,211832,211893]]


# lstm_matrix = metrics.confusion_matrix(labels,lstm_labels)
# pd.DataFrame(lstm_matrix, columns = ['Normal', 'Anomalies'], index= ['Normal', 'Anomalies']).to_csv('lstm_matrix.csv')

# cnn_matrix = metrics.confusion_matrix(labels,cnn_labels)
# pd.DataFrame(cnn_matrix, columns = ['Normal', 'Anomalies'], index= ['Normal', 'Anomalies']).to_csv('cnn_matrix.csv')

# hybrid_matrix = metrics.confusion_matrix(labels,hybrid_labels)
# pd.DataFrame(hybrid_matrix, columns = ['Normal', 'Anomalies'], index= ['Normal', 'Anomalies']).to_csv('hybrid_matrix.csv')



######plot errors vertical
# def to_dt3(prediction_data,prediction_anomalies, indx, save = False, ae = ''):
#     noise = 91.2
#     pressure = 53.98
#     columns = ['Speed','Noise','Pressure']
#     columns2 = ['Speed_anomalies','Noise_anomalies','Pressure_anomalies']
#     index = np.asarray(list(range(201))) * 0.1


#     predictions = dc(prediction_data[indx])
#     tst = dc(test[indx])
#     anomalies = dc(prediction_anomalies[indx])
#     dummy = dc(dummys_array[indx])

#     predictions[:,1] = predictions[:,1]*91.2
#     tst[:,1] = tst[:,1]*91.2
#     predictions[:,2] = predictions[:,2]*53.98
#     tst[:,2] = tst[:,2]*53.98

#     anomalies[:,1] = anomalies[:,1]*91.2
#     dummy[:,1] = dummy[:,1]*91.2
#     anomalies[:,2] = anomalies[:,2]*53.98
#     dummy[:,2] = dummy[:,2]*53.98

#     predictions[predictions<0] = 0
#     anomalies[anomalies<0] = 0

#     error_speed_normal = metrics.mean_squared_error(predictions[:,0], tst[:,0])
#     error_noise_normal = metrics.mean_squared_error(predictions[:,1], tst[:,1])
#     error_pressure_normal = metrics.mean_squared_error(predictions[:,2], tst[:,2])

#     error_speed_anomal = metrics.mean_squared_error(anomalies[:,0], dummy[:,0])
#     error_noise_anormal = metrics.mean_squared_error(anomalies[:,1], dummy[:,1])
#     error_pressure_anormal = metrics.mean_squared_error(anomalies[:,2], dummy[:,2])
    
#     increase_speed = int(((error_speed_anomal-error_speed_normal)/error_speed_normal)*100)
#     increase_noise = int(((error_noise_anormal-error_noise_normal)/error_noise_normal)*100)
#     increase_pressure = int(((error_pressure_anormal-error_pressure_normal)/error_pressure_normal)*100)

#     df_pred = pd.DataFrame(predictions,columns=columns, index = index)
#     df_real = pd.DataFrame(tst,columns=columns, index = index)
#     df_pred_a = pd.DataFrame(anomalies,columns=columns2, index = index)
#     df_real_a = pd.DataFrame(dummy,columns=columns2, index = index)

#     df_pred = pd.concat([df_pred,df_pred_a], axis = 1)
#     df_real = pd.concat([df_real,df_real_a], axis = 1)


#     dist_sp = df_real['Speed'].loc[np.abs(df_real['Speed'] - df_pred['Speed']) > df_pred['Speed'].max() * 0.15]
#     dist_noise = df_real['Noise'].loc[df_real['Noise'] > 69]
#     dist_pressure = df_real['Pressure'].loc[np.abs(df_real['Pressure'] - df_pred['Pressure']) > 1.9]

#     dist_sp2 = df_real_a['Speed_anomalies'].loc[np.abs(df_real_a['Speed_anomalies'] - df_pred_a['Speed_anomalies']) > df_pred_a['Speed_anomalies'].max() * 0.15]
#     dist_noise2 = df_real_a['Noise_anomalies'].loc[df_real_a['Noise_anomalies'] > 69]
#     dist_pressure2 = df_real_a['Pressure_anomalies'].loc[np.abs(df_real_a['Pressure_anomalies'] - df_pred_a['Pressure_anomalies']) > 1.9]
    
#     li = [dist_sp,dist_sp2,dist_noise,dist_noise2,dist_pressure, dist_pressure2]

#     errors = [error_speed_normal, error_speed_anomal, error_noise_normal, error_noise_anormal, error_pressure_normal, error_pressure_anormal]

#     increases = ['',increase_speed,'',increase_noise,'', increase_pressure]

#     fig, axes = plt.subplots(figsize=(9,13), nrows=3, ncols=2, sharex=True, tight_layout= True)
#     plot_columns = ['Speed', 'Speed with\nanomalies', 'Noise','Noise with\nanomalies', 'Pressure',  'Pressure with\nanomalies']
#     axes = axes.ravel(order='C')
#     plot_sequence = ['Speed','Speed_anomalies','Noise','Noise_anomalies','Pressure', 'Pressure_anomalies' ]
#     plot_y_labels = ['m/s','m/s', 'dB','dB', 'bar','bar']
#     plot_x_labels = ['time(s)','time(s)', 'time(s)','time(s)', 'time(s)','time(s)']
#     for i in range(6):
#         axes[i].plot(df_real[plot_sequence[i]], label ='Actual', color = 'b', linewidth = 3)
#         axes[i].plot(df_pred[plot_sequence[i]], label ='Reconstruction', color = 'g')
#         if i %2 ==0:
#             axes[i].annotate('MAE = {}'.format(np.round(errors[i], 7)), xy=(.58,.22),xycoords = 'axes fraction')
#         else:
#             axes[i].annotate('MAE = {}\nError increased {}%'.format(np.round(errors[i], 7), increases[i]), xy=(.56,.22),xycoords = 'axes fraction')
#             axes[i].get_yaxis().set_visible(False)
#         axes[i].set_title(plot_columns[i])
#         axes[i].fill_between(df_pred.index, df_real[plot_sequence[i]],df_pred[plot_sequence[i]], color = 'gray', label = 'Error')
#         #axes[i].plot(li[i], 'x', label = 'Above_Thresholds', color = 'r')
#         axes[i].legend(loc = 'upper right')
#         axes[i].set_ylabel(plot_y_labels[i])
#         if i ==4 or i ==5: 
#             axes[i].set_xlabel(plot_x_labels[i])
#     plt.suptitle(ae, fontsize = 19)
#     if save:
#         plt.savefig('{}{}error.png'.format(indx,ae))
#     plt.show()



#######example order
# fig, axes = plt.subplots(figsize=(7,6), nrows=6, ncols=1, tight_layout= True, sharex = True)

# dtfrm.loc[orders[0]][['TIME','SPEED_UP']].plot(x = 'TIME',ax = axes[0])
# axes[0].set_ylabel('m/s')
# axes[0].legend(['Speed up'], fontsize = 11)

# dtfrm.loc[orders[0]][['TIME','NOISE_UP']].plot(x =  'TIME',ax = axes[1], color = 'orange')
# axes[1].set_ylabel('dB')
# axes[1].legend(['Noise up'], fontsize = 11)

# dtfrm.loc[orders[0]][['TIME','PRESSURE_UP']].plot(x =  'TIME',ax = axes[2], color = 'g')
# axes[2].set_ylabel('bar')
# axes[2].legend(['Pressure up'], fontsize = 11)

# dtfrm.loc[orders[0]][['TIME','SPEED_DOWN']].plot(x = 'TIME',ax = axes[3], color = 'r')
# axes[3].set_ylabel('m/s')
# axes[3].legend(['Speed down'], fontsize = 11)

# dtfrm.loc[orders[0]][['TIME','NOISE_DOWN']].plot(x =  'TIME',ax = axes[4], color = 'm')
# axes[4].set_ylabel('dB')
# axes[4].legend(['Noise down'], fontsize = 11)

# dtfrm.loc[orders[0]][['TIME','PRESSURE_DOWN']].plot(x =  'TIME',ax = axes[5], color = 'brown')
# axes[5].set_ylabel('bar')
# axes[5].legend(['Pressure down'], fontsize = 11)
# axes[5].set_xlabel('time(s)')



# plt.savefig('example_order')