import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, matthews_corrcoef

# matplotlib.use('WX')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.dates as mdates
import matplotlib.lines as mlines


def newline(p1, p2, l_width=0.3, color='lightgray'):
    ax = plt.gca()
    l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], linewidth=l_width, color=color)
    ax.add_line(l)
    return l


df = pd.read_csv('/Users/alessandro/Documents/WD_references_analysis/results/train_set_references.csv')
df_properties = df['stat_property']
df_properties = df_properties.to_frame()
df_properties.reset_index(level=0, inplace=True)

predicted_authoritative = pd.read_csv('/Users/alessandro/Documents/WD_references_analysis/predicted_authoritative.csv')
predicted_authoritative.columns = ['index', 'expected', 'predicted']
predicted_authoritative.sort_values(by='index', inplace=True)
predicted_authoritative['index'].drop_duplicates(keep='first', inplace=True)

predicted_authoritative = predicted_authoritative.merge(df_properties, on='index')


def check_metrics(row):
    expected = row['expected']
    predicted = row['predicted']
    precision = precision_score(expected, predicted, average='weighted', pos_label=1)
    recall = recall_score(expected, predicted, average='weighted', pos_label=1)
    f1 = f1_score(expected, predicted, average='weighted', pos_label=1)
    auc_pr = average_precision_score(expected, predicted, average='weighted')
    mcc = matthews_corrcoef(expected, predicted)
    return precision, recall, f1, auc_pr, mcc


predicted_by_property = predicted_authoritative.groupby('stat_property').apply(check_metrics)
predicted_by_property = predicted_by_property.to_frame()
predicted_by_property.reset_index(level=0, inplace=True)
predicted_by_property.columns = ['stat_property', 'metrics']

predicted_by_property = pd.merge(predicted_by_property, pd.DataFrame(predicted_by_property['metrics'].tolist(),
                                                                     index=predicted_by_property.index),
                                 left_index=True, right_index=True)
predicted_by_property.drop('metrics', axis=1, inplace=True)
predicted_by_property.columns = ['stat_property', 'precision', 'recall', 'f1', 'auc_pr', 'mcc']

# add property count
property_counts = df_properties.stat_property.value_counts().to_frame().reset_index()
property_counts.columns = ['stat_property', 'usage_count']

predicted_by_property = predicted_by_property.merge(property_counts, on='stat_property')
# .loc[predicted_by_property['usage_count'] >= 5, ]
predicted_by_property.sort_values(by='usage_count', ascending=False, inplace=True)

predicted_by_property['serial'] = range(0, predicted_by_property.shape[0])
predicted_by_property_50 = predicted_by_property.loc[predicted_by_property['serial'] <= 49,]

###number of uses

fig = plt.figure(figsize=(20, 10))

ax = plt.subplot(121)
ax.vlines(x=0.2, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax.vlines(x=0.2, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax.vlines(x=0.4, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax.vlines(x=0.6, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax.vlines(x=0.8, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')

ax.hlines(y=predicted_by_property_50['serial'], xmin=0, xmax=1, color='lightgray', linewidth=0.2, linestyles='--')


for i, p2 in zip(predicted_by_property_50['serial'], predicted_by_property_50['f1']):
    newline([0, i], [p2, i], l_width=0.4, color='#0c374d')

ax.scatter(y=predicted_by_property_50['serial'], x=predicted_by_property_50['f1'], s=40, color='#0c374d')

ax.set_yticks(predicted_by_property_50['serial'])
ax.set_yticklabels(predicted_by_property_50['stat_property'], fontdict={'horizontalalignment': 'right'})
ax.invert_yaxis()

ax2 = plt.subplot(122)
# ax2.vlines(x=-0.8, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax2.vlines(x=-0.75, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax2.vlines(x=-0.50, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax2.vlines(x=-0.25, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax2.vlines(x=-0, ymin=0, ymax=49, color='lightgray', linewidth=0.6)
ax2.vlines(x=0.25, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax2.vlines(x=0.50, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax2.vlines(x=0.75, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
# ax2.vlines(x=0.8, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')


ax2.hlines(y=predicted_by_property_50['serial'], xmin=-1, xmax=1, color='lightgray', linewidth=0.2, linestyles='--')

ax2.scatter(y=predicted_by_property_50['serial'], x=predicted_by_property_50['mcc'], s=40, color='#eb4034')
for i, p2 in zip(predicted_by_property_50['serial'], predicted_by_property_50['mcc']):
    newline([0, i], [p2, i], l_width=0.4, color='#eb4034')

ax2.set_yticks([])
ax2.invert_yaxis()

plt.yticks(fontsize=11.7)
plt.tight_layout()
plt.show()
plt.savefig('f1_mcc_by_property.eps', format='eps', transparent=True)


### Number of uses


fig = plt.figure(figsize=(20, 10))

ax = plt.subplot(121)
ax.vlines(x=-0.75, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax.vlines(x=-0.50, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax.vlines(x=-0.25, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax.vlines(x=-0, ymin=0, ymax=49, color='lightgray', linewidth=0.6)
ax.vlines(x=0.25, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax.vlines(x=0.50, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax.vlines(x=0.75, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')

ax.hlines(y=predicted_by_property_50['serial'], xmin=-1, xmax=1, color='lightgray', linewidth=0.2, linestyles='--')

# for i, p2 in zip(predicted_by_property_50['serial'], predicted_by_property_50['f1']):
#     newline([0, i], [p2, i], l_width=0.4, color='#0c374d')


ax.scatter(y=predicted_by_property_50['serial'], x=predicted_by_property_50['f1'], s=40, color='#0c374d')
ax.scatter(y=predicted_by_property_50['serial'], x=predicted_by_property_50['mcc'], s=40, color='#eb4034')


ax.set_yticks(predicted_by_property_50['serial'])
ax.set_yticklabels(predicted_by_property_50['stat_property'], fontdict={'horizontalalignment': 'right'})
ax.invert_yaxis()

ax2 = plt.subplot(122)
# ax2.vlines(x=-0.8, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax2.vlines(x=-0.75, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax2.vlines(x=-0.50, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax2.vlines(x=-0.25, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax2.vlines(x=-0, ymin=0, ymax=49, color='lightgray', linewidth=0.6)
ax2.vlines(x=0.25, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax2.vlines(x=0.50, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
ax2.vlines(x=0.75, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')
# ax2.vlines(x=0.8, ymin=0, ymax=49, color='lightgray', linewidth=0.3, linestyles='--')


ax2.hlines(y=predicted_by_property_50['serial'], xmin=-1, xmax=1, color='lightgray', linewidth=0.2, linestyles='--')

ax.scatter(y=predicted_by_property_50['serial'], x=predicted_by_property_50['f1'], s=40, color='#0c374d')
ax2.scatter(y=predicted_by_property_50['serial'], x=predicted_by_property_50['mcc'], s=40, color='#eb4034')
# for i, p2 in zip(predicted_by_property_50['serial'], predicted_by_property_50['mcc']):
#     newline([0, i], [p2, i], l_width=0.4, color='#eb4034')

ax2.set_yticks([])
ax2.invert_yaxis()

plt.yticks(fontsize=11.7)
plt.tight_layout()
plt.show()
plt.savefig('f1_mcc_by_property.eps', format='eps', transparent=True)
