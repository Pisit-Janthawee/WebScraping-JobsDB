import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind

class Viz:
    def __init__(self, palette_name='YlOrBr'):
        self.palette_name = palette_name
        self.palette = sns.color_palette(palette_name)
        self.binary_palette = sns.color_palette(palette_name, 2)
        self.colors = {
            'plotly': px.colors.qualitative.Set1,
            'seaborn': sns.color_palette(palette_name),
            # Add more color mappings as needed
        }
        sns.set_style("darkgrid")

    def skewness_boxplots(self, data, columns, n_rows, n_cols, suptitle):
        num_plots = n_rows * n_cols

        if num_plots < len(columns):
            raise ValueError("Number of subplots is less than the number of columns in columns_list.")

        fig, axs = plt.subplots(n_rows, n_cols, sharey=True, figsize=(16, 25))
        fig.suptitle(suptitle, y=1, size=25)
        axs = axs.flatten()

        for i, col in enumerate(columns):
            sns.boxplot(data=data[col], orient='h', ax=axs[i])
            axs[i].set_title(col + ', skewness is: ' +
                            str(round(data[col].skew(axis=0, skipna=True), 2)))

        plt.tight_layout()
        plt.show()
    def boxplots_with_target(self, data, features, target, n_cols=4):
        num_features = len(features)
        num_rows = (num_features + n_cols - 1) // n_cols  # Calculate the number of rows needed
        fig, axes = plt.subplots(nrows=num_rows, ncols=n_cols, figsize=(15, 5 * num_rows))

        for i, feature_column in enumerate(features):
            row_idx = i // n_cols
            col_idx = i % n_cols

            # Create horizontal boxplot
            sns.boxplot(x=data[target], y=data[feature_column], ax=axes[row_idx, col_idx], palette=self.palette_name)
            axes[row_idx, col_idx].set_title(f'Boxplot for {feature_column} by {target}')
            axes[row_idx, col_idx].set_xlabel(feature_column)
            axes[row_idx, col_idx].set_ylabel(target)

        # Hide empty subplots
        for i in range(num_features, num_rows * n_cols):
            row_idx = i // n_cols
            col_idx = i % n_cols
            fig.delaxes(axes[row_idx, col_idx])

        plt.tight_layout()
        plt.show()
    def box_plot_horizontal(self, data, columns):
        # Melt the DataFrame to long-form
        melted_df = pd.melt(data[columns])

        box_plot = sns.boxplot(x='variable', y='value', data=melted_df)
        sns.set(rc={'figure.figsize': (20, 6)})
        ax = box_plot.axes
        lines = ax.get_lines()
        categories = ax.get_xticks()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

        for cat in categories:
            y = round(lines[4 + cat * 6].get_ydata()[0], 1)

            ax.text(
                cat,
                y,
                f'{y}',
                ha='center',
                va='center',
                fontweight='bold',
                size=7,
                color='white',
                bbox=dict(facecolor='#445A64'))
        plt.title('Box Plot', fontsize=20)
        plt.show()
    
    def plot_trend(self, data, x, y, target, title, n_cols=4):
        columns = x
        n_rows = int(np.ceil(len(columns) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))

        for i, col in enumerate(columns):
            ax = axes[i // n_cols, i % n_cols]
            sns.lineplot(x=col, y=y, hue=target, data=data,
                        marker='o', markersize=5, ax=ax,palette=self.palette_name)

            # Add trend line using sns.regplot
            sns.regplot(x=col, y=y, data=data, scatter=False, ax=ax, color='gray', ci=None, line_kws={'linestyle': '--'})

            ax.set_ylabel(y)
            ax.set_xlabel(f'{col}')
            ax.legend()
            ax.set_title(f'{y} in ' + r'$\bf{' + f'{col}' + '}$' + ' Trend')

            # Adding annotations
            y_values = data[y]
            avg_value = y_values.mean()  # Calculate the average value

            step = int(len(data[col]) / 10)
            for j in range(0, len(y_values), step):
                ax.annotate(f'{y_values.iloc[j]:.2f}',
                            (y_values.iloc[j], data.loc[j, y]), arrowprops=dict(arrowstyle='->'))

            # Add average value as text
            ax.text(0.95, 0.95, f'Average: {avg_value:.2f}', transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        plt.suptitle(f"{title}", fontsize=16, fontweight='bold')
        plt.legend(title=target)
        fig.set_facecolor('white')
        plt.tight_layout()


    def pearson_correlation(self, data, figsize, mode='full', target=None):
        if mode == 'full':
            fig, axes = plt.subplots(figsize=figsize)
            plt.title('Pearson Correlation Matrix', fontsize=20)
            sns.heatmap(data.select_dtypes(include=[np.number]).corr(), linewidths=0, vmax=0.7, square=True, cmap=self.palette_name,
                        linecolor='w', annot=True, annot_kws={"size": 10}, cbar_kws={"shrink": .9})
            plt.tight_layout()
            plt.show()
        elif mode == 'target':
            fig, axes = plt.subplots(figsize=(20, 6))
            fig.suptitle('Pearson Correlation Matrix',
                         fontsize=20, fontweight='bold')

            corr_matrix = data.select_dtypes(include=[np.number]).corr()
            # Df section
            target_corr = corr_matrix[target]
            other_corr = corr_matrix.drop(target)[target]
            corr_df = pd.DataFrame({target: target_corr})
            corr_df = corr_df.reindex(
                corr_df[target].abs().sort_values(ascending=False).index)
            # Graph Section

            axes.set_title(r'$\bf{' + 'Original' + '}$' +
                           f'\n{target}')
            sns.heatmap(corr_df, annot=True, ax=axes, cmap=self.palette_name)
            plt.vlines(x=1, ymin=0, ymax=len(corr_df), colors='r', linewidth=2)
            plt.show()
            return corr_df
        else:
            raise ValueError(
                'Invalid mode. Mode should be either "full" or "target".')
        

    def histogram_boxplot(self, data, target, target_classes, plotting_library='plotly',):
        for col in data.select_dtypes(include=[np.number]):
            min_ = round(data[col].min(), 2)
            mean_ = round(data[col].mean(), 2)
            median_ = round(data[col].median(), 2)
            mode_ = data[col].mode()[0]
            max_ = round(data[col].max(), 2)
            std_ = round(data[col].std(), 2)
            skewness = round(data[col].skew(axis=0, skipna=True), 2)

            total_shape = data.shape[0]
            col_shape = data[col].loc[data[col].notna()].shape[0]

            class_0 = f'{target_classes[0]} {round(data[col][data[target] == 0].shape[0]/ len(data) * 100, 2)}'+'%'
            class_1 = f'{target_classes[1]} {round( data[col][data[target] == 1].shape[0] / len(data) * 100, 2)}'+'%'

            if plotting_library == 'plotly':
                fig = px.histogram(data, x=col, marginal='box',
                                   color=target, color_discrete_sequence=self.colors[plotting_library])
                fig.update_traces(marker=dict(
                    line=dict(width=0, color='Black')))
                fig.update_layout(bargap=0.1)

                fig.update_layout(
                    title_text=f'<b>{col}</b> ({col_shape} of {total_shape})<br>{class_0} {class_1}<br><b>Min: {min_}, Mean:{mean_}, Median:{median_}, Mode:{mode_}, Max:{max_}, Std:{std_}, \nSkewness: {skewness}</b>',
                )
                fig.show()

            elif plotting_library == 'seaborn':
                fig, axes = plt.subplots(1, 2, figsize=(15, 9))
                sns.histplot(data=data, x=col, hue=target, element='step',
                             fill=True, kde=True, stat='density', ax=axes[0], palette=self.palette_name)
                sns.boxplot(data=data, x=col, y=target,
                            orient='h', palette=self.palette_name, ax=axes[1])

                fig.suptitle(
                    f'{col} ({col_shape} of {total_shape})\n{class_0} {class_1}\nMin: {min_}, Mean: {mean_}, Median: {median_}, Mode: {mode_}, Max: {max_}, Std: {std_}, </b>Skewness: {skewness}')
                plt.show()

    def pie_plot(self, data, columns, percentage_option='both', top_n=10, num_cols=2):
        num_features = len(columns)
        rows = num_features // num_cols if num_features % num_cols == 0 else num_features // num_cols + 1

        plt.figure(figsize=(20, 6 * rows))

        for i, col in enumerate(columns):
            plt.subplot(rows, num_cols, i + 1)
            value_counts = data[col].value_counts()
            sorted_value_counts = value_counts.sort_values(ascending=False)
            top_values = sorted_value_counts.head(top_n)
            labels = top_values.index
            sizes = top_values.values
            # Determine the format for autopct based on percentage_option
            if percentage_option == 'number':
                autopct_format = lambda p:  f'{int(p * sum(sizes) / 100)}'
            elif percentage_option == 'percentage':
                autopct_format = '%1.1f%%'
            else:  # 'both'
                autopct_format = lambda p: f'{int(p * sum(sizes) / 100)}\n{p:.1f}%'

            plt.pie(sizes, labels=labels, autopct=autopct_format, startangle=90,colors=sns.color_palette(self.palette_name))
            plt.title(f'Top {top_n} values in {col}')

        plt.tight_layout()
        plt.show()

    def bar_plot_top_values(self, data, columns, display_percentage=False, top_n=10, num_cols=2):
        n_cols = len(columns)
        n_rows = int(np.ceil(n_cols / num_cols))

        plt.figure(figsize=(20, 6 * n_rows))

        for i, col in enumerate(columns):
            plt.subplot(n_rows, num_cols, i + 1)
            value_counts = data[col].value_counts()
            sorted_value_counts = value_counts.sort_values(ascending=False)
            top_values = sorted_value_counts.head(top_n)

            ax = sns.barplot(x=top_values.index,
                             y=top_values.values, dodge=False, palette=self.palette_name)
            plt.title(f'Top {top_n} values in {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            if display_percentage:
                total = sum(value_counts)
                for index, value in enumerate(top_values):
                    percentage = (value / total) * 100
                    ax.text(index, value, f'{percentage:.2f}%',
                            ha='center', va='bottom')
            else:
                for index, value in enumerate(top_values):
                    ax.text(index, value, str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    

    def plot_histogram(self, data, target, title, n_cols=4, bins=100):
        columns = data.select_dtypes(include=[np.number]).columns
        n_rows = int(np.ceil(len(columns) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))

        for i, col in enumerate(columns):
            OC = OutlierCleaner()
            min_, mean_, median_, mode_, max_, std_, q1, q3, iqr, lower_bound, upper_bound, distinct_count, shape = OC.summary_stats(
                data, col, None)
            percent_missing = round(100 - (shape / data.shape[0]) * 100, 3)
            
            skewness_before = round(data.select_dtypes(include=np.number)[
                                    col].skew(axis=0, skipna=True), 2)
            legend_labels = data[target].unique().tolist()[::-1]
            sns.histplot(data=data, x=col, hue=target, common_norm=False,
                         stat='density', multiple='dodge', ax=axes[i // n_cols, i % n_cols], bins=bins, kde=True, element='step', fill=True,
                         palette=self.palette_name)
            axes[i // n_cols, i % n_cols].set_title(

                r'$\bf{'+f'{col}'+'}$' + f'\n{shape} of {data.shape[0]} (Missing = {data.shape[0]-shape} ({percent_missing}%)\n' +
                f'Min: {min_}, Mean: {mean_}, Median: {median_}, Mode: {mode_}, Std: {std_}, Max: {max_}, Distinct: {distinct_count}\nQ1-Q3 ({q1} - {q3})\nSkewness: {skewness_before}')

            axes[i // n_cols, i % n_cols].set_xlabel(f'{col}')
            axes[i // n_cols, i %
                 n_cols].legend(labels=legend_labels)

        plt.suptitle(f"{title}", fontsize=16, fontweight='bold')

        fig.set_facecolor('white')
        plt.tight_layout()
    def perform_t_test_and_plot(self, data, features, target, n_cols, alpha=0.05):
        num_features = len(features)
        num_rows = (num_features + n_cols - 1) // n_cols  # Calculate the number of rows needed
        fig, axes = plt.subplots(nrows=num_rows, ncols=n_cols, figsize=(15, 5 * num_rows))
        results = []

        for i, feature_column in enumerate(features):
            row_idx = i // n_cols
            col_idx = i % n_cols

            # Separate data into two groups based on the target variable
            group_0 = data[data[target] == 0][feature_column].dropna()
            group_1 = data[data[target] == 1][feature_column].dropna()

            # Perform t-test for independent samples
            t_statistic, p_value = ttest_ind(group_0, group_1, equal_var=False)

            # Determine whether the feature is statistically significant
            if p_value < alpha:
                significance_text = f'Reject Null Hypothesis (alpha={alpha}) Useful'
            else:
                significance_text = f'Accept Null Hypothesis (alpha={alpha})'

            # Plot the boxplot on the (row_idx, col_idx)-th subplot
            sns.boxplot(x=data[target], y=data[feature_column], ax=axes[row_idx, col_idx], palette=self.palette_name)
            axes[row_idx, col_idx].set_title(f'T-Test Results for {feature_column} and {target}\n'
                                            f'T-Statistic: {t_statistic:.2f}, P-Value: {p_value:.4f}{significance_text}')
            # Append results to the DataFrame
            results.append({
                'Feature': feature_column,
                'T-Statistic': t_statistic,
                'P-Value': p_value,
                'Significance': significance_text
            })
        # Hide empty subplots
        for i in range(num_features, num_rows * n_cols):
            row_idx = i // n_cols
            col_idx = i % n_cols
            fig.delaxes(axes[row_idx, col_idx])

        plt.tight_layout()
        plt.show()
        return pd.DataFrame(results)