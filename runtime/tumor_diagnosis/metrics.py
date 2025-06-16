from digicare.utilities.database import Database
import numpy as np
from sklearn import metrics
from digicare.utilities.file_ops import join_path, mkdir, gd
from digicare.utilities.plot import multi_curve_plot
from digicare.utilities.data_io import SimpleExcelWriter

def get_prof_stat_db_keys():
    return [
        'data_source', # aka "data_center", they are the same.
        'subject_name',
        'real_class',
        'pred_class_probs',
        'grad_cam',
    ]

def create_database_for_performance_statistics(xlsx_file=None):
    return Database(get_prof_stat_db_keys(), xlsx_file=xlsx_file)

class PerformanceStatistics:
    def __init__(self, all_class_names):
        self.all_class_names = all_class_names
        self.reset()

    def reset(self):
        self.database = create_database_for_performance_statistics()

    def record(self, subject_name, data_source, pred_class_probs, real_class, grad_cam_output):
        assert subject_name not in self.database.data_dict['subject_name'], \
            'Error, subject "%s" is already recorded, cannot record it twice!' % subject_name
        assert real_class in self.all_class_names, 'Unrecognized class "%s" given. '\
            'Acceptable classes are: "%s".' % (real_class, str(self.all_class_names))
        record = self.database.make_empty_record()
        record['data_source']      = data_source
        record['subject_name']     = subject_name
        record['pred_class_probs'] = str(pred_class_probs)
        record['real_class']       = real_class
        record['grad_cam']         = grad_cam_output
        self.database.add_record(record)

    def save_to_xlsx(self, output_xlsx):
        mkdir(gd(output_xlsx))
        self.database.export_xlsx(output_xlsx, left_freezed_cols=2)
    
    def load_from_xlsx(self, input_xlsx):
        self.reset()
        self.database = create_database_for_performance_statistics(xlsx_file=input_xlsx)
    
    def _calculate_confusion_matrix_metrics_wrt_thresholds(self, database: Database, positive_class_name: str):
        assert positive_class_name in self.all_class_names, \
            'Positive class name "%s" is not in "%s".' % (positive_class_name, str(self.all_class_names))
        def _calculate_confusion_matrix_wrt_threshold(database: Database, positive_class_name: str, threshold: float):
            num_records = database.num_records()
            confusion_matrix = {'tp':0, 'tn':0, 'fp': 0, 'fn': 0}
            for record_id in range(num_records):
                record = database.get_record(record_id)
                pred_class_probs_dict = eval(record['pred_class_probs'])
                real_class = record['real_class']
                all_pred_classes = list(pred_class_probs_dict.keys())
                assert real_class in all_pred_classes, 'Error, real class "%s" is not included in all possible classes "%s".' % (real_class, all_pred_classes)
                assert len(all_pred_classes) == 2, 'Number of classes should = 2, got %d.' % len(all_pred_classes)
                assert 'yes' in all_pred_classes and 'no' in all_pred_classes
                other_class_name = all_pred_classes[0] if all_pred_classes.index(positive_class_name) == 1 else all_pred_classes[1]
                pred_class = positive_class_name if pred_class_probs_dict[positive_class_name] > threshold else other_class_name
                if   pred_class == 'yes' and real_class == 'yes': confusion_matrix['tp'] += 1
                elif pred_class == 'yes' and real_class == 'no':  confusion_matrix['fp'] += 1
                elif pred_class == 'no'  and real_class == 'yes': confusion_matrix['fn'] += 1
                elif pred_class == 'no'  and real_class == 'no':  confusion_matrix['tn'] += 1
                else: raise RuntimeError('unknown pred and real combination.')
            return confusion_matrix
        def _scan_threshold_and_calculate_acc(database: Database, positive_class_name: str):
            thresholds = []
            acc_wrt_thresholds = []
            for prob_threshold in list(np.linspace(0,1,100)):
                thresholds.append(prob_threshold)
                cm = _calculate_confusion_matrix_wrt_threshold(database, positive_class_name,prob_threshold)
                acc = (cm['tp'] + cm['tn']) / (cm['tp'] + cm['tn'] + cm['fp'] + cm['fn'])
                acc_wrt_thresholds.append(acc)
            return thresholds, acc_wrt_thresholds
        thresholds, acc_wrt_thresholds = _scan_threshold_and_calculate_acc(database, positive_class_name)
        confusion_matrix = _calculate_confusion_matrix_wrt_threshold(database, positive_class_name, 0.5)
        tp = confusion_matrix['tp']
        tn = confusion_matrix['tn']
        fp = confusion_matrix['fp']
        fn = confusion_matrix['fn']
        print('TP=%d, TN=%d, FP=%d, FN=%d' % (tp, tn, fp, fn))
        acc = (tp + tn) / (tp + tn + fp + fn)                if tp + tn + fp + fn != 0 else 'N/A'
        precision = (tp) / (tp + fp)                         if tp + fp != 0           else 'N/A'
        recall = sensitivity = tpr = (tp) / (tp + fn)        if tp + fn != 0           else 'N/A'
        f1 = (2 * precision * recall) / (precision + recall) if tp != 0                else 'N/A'
        specificity = tnr = (tn) / (tn + fp)                 if tn + fp != 0           else 'N/A'
        fpr = (fp) / (fp + tn)                               if fp + tn != 0           else 'N/A'
        fnr = (fn) / (fn + tp)                               if fn + tp != 0           else 'N/A'
        acc_wrt_thresholds_curve = {
            'ACC' : { 'x': thresholds, 'y': acc_wrt_thresholds,  'color': [1.0, 0.0, 0.0], 'label': True},
        }
        # mkdir(gd(save_file))
        # multi_curve_plot(curves, save_file, dpi=150, ylim=[0,1], title='ACC metric when different thresholds applied', 
        #                     xlabel='Accept class "%s" if its probability is greater than threshold' % positive_class_name, ylabel='ACC(%)')
        return acc, precision, recall, f1, sensitivity, specificity, tpr, tnr, fpr, fnr, acc_wrt_thresholds_curve

    def _calculate_roc_auc_metrics(self, database: Database):
        '''
        return AUC and ROC curve points.
        '''
        print('* calculating roc auc metric...')
        all_classes = list(set(['yes', 'no'] + database.data_dict['real_class']))
        assert len(all_classes) == 2, 'must be 2 classes ("yes" and "no"), got one or more unknown class %s.' % \
            str([clsname for clsname in all_classes if clsname not in ['yes', 'no']])
        assert 'yes' in all_classes and 'no' in all_classes, 'class type should be "yes" and "no".'
        y, pred = [], []
        for i in range(database.num_records()):
            record = database.get_record(i)
            pred_class_probs_dict = eval(record['pred_class_probs'])
            pred_positive_prob = pred_class_probs_dict['yes']
            real_class = record['real_class']
            label_value = 1 if real_class == 'yes' else 0
            y.append(label_value)
            pred.append(pred_positive_prob)
        y, pred = np.array(y), np.array(pred)
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # sort fpr tpr
        roc_points = []
        for fpr_, tpr_ in zip(fpr, tpr):
            roc_points.append((fpr_, tpr_)) # (x, y)
        roc_points = sorted(roc_points, key=lambda x:x[0])
        if np.isnan(auc): auc = 'N/A'
        print('AUC=%s' % (auc if isinstance(auc, str) else '%.4f' % auc))
        roc_curve = {
            'ROC' : { 'x': [p[0] for p in roc_points], 'y': [p[1] for p in roc_points],  'color': [1.0, 0.0, 0.0], 'label': True},
        }
        return roc_curve, auc

    def measure_performance_binary(self, output_folder):
        '''
        Measure the performance of a binary classifier.
        '''
        if len(self.all_class_names) != 2:
            print('[*] not a binary classification problem.')
            return
        
        assert 'yes' in self.all_class_names and 'no' in self.all_class_names, \
            'Expect both "yes" and "no" in class names but failed.'

        all_data_centers = ['ALL'] + list(set(self.database.data_dict['data_source']))
        output_xlsx = join_path(output_folder, 'scalar_metrics.xlsx')
        mkdir(output_folder)
        xlsx_obj = SimpleExcelWriter(output_xlsx, worksheet_names=all_data_centers)

        def _measure_performance_for_each_DC(data_center_name: str, xlsx_obj: SimpleExcelWriter):
            # measure performance for each data center and save results into xlsx,
            # returns acc_curve and roc_curve as they cannot be saved simply in a xlsx.
            # step 1: filter record
            database_for_this_DC = create_database_for_performance_statistics()
            for record_id in range(self.database.num_records()):
                record = self.database.get_record(record_id)
                if record['data_source'] == data_center_name or data_center_name == 'ALL':
                    database_for_this_DC.add_record(record)            
            # step 2: calculate all metrics related to confusion matrix
            acc, precision, recall, f1, sensitivity, specificity, \
            tpr, tnr, fpr, fnr, acc_wrt_thresholds_curve = \
                self._calculate_confusion_matrix_metrics_wrt_thresholds(database_for_this_DC, "yes")
            roc_curve, auc = self._calculate_roc_auc_metrics(database_for_this_DC)
            # step 3: write to xlsx
            top_fmt = xlsx_obj.new_format(font_color='#FFFFFF', bg_color='#606060', bold=True)
            col_fmt = xlsx_obj.new_format(font_color='#000000', bg_color='#D9D9D9')
            xlsx_obj.write((0,0), 'Metric', format=top_fmt, worksheet_name=data_center_name)
            xlsx_obj.write((0,1), 'Value',  format=top_fmt, worksheet_name=data_center_name)
            packed_metrics = [acc, precision, recall, f1, sensitivity, specificity, tpr, tnr, fpr, fnr, auc]
            packed_metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'Sensitivity', 'Specificity', 
                 'True Positive Rate (TPR)', 'True Negative Rate (TNR)', 'False Positive Rate (FPR)',
                 'False Negative Rate (FNR)', 'AUROC']
            for metric_name, metric_value, metric_id in zip(packed_metric_names, packed_metrics, range(len(packed_metrics))):
                xlsx_obj.write((metric_id+1, 0), metric_name, format=col_fmt, worksheet_name=data_center_name)
                xlsx_obj.write((metric_id+1, 1), ('%.4f' % metric_value) if isinstance(metric_value, float) else str(metric_value), 
                               worksheet_name=data_center_name)
            xlsx_obj.set_zoom(85,worksheet_name=data_center_name)
            return acc_wrt_thresholds_curve, roc_curve

        acc_curves, roc_curves = {}, {}
        curve_color_palette = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1,0.5,0], [1,0,0.5], [0.5,1,0],
            [0,1,0.5], [0.5,0,1], [0,0.5,1],
            [1,1,0.5], [1,0.5,1], [0.5,1,1],
            [0.5,0.5,1], [0.5,1,0.5], [1,0.5,0.5],
            [0, 1, 1], [1, 0, 1], [1, 1, 0],
            [0, 0.25, 1], [0.25, 0, 1], [0.25, 1, 0], [0, 1, 0.25], [1, 0.25, 0], [1, 0, 0.25], 
        ]
        for data_center_name, data_center_id in zip(all_data_centers, range(len(all_data_centers))):
            print('Data source:', data_center_name)
            if data_center_id >= len(curve_color_palette):
                raise RuntimeError('Tool much data centers!')
            curve_color_for_this_DC = curve_color_palette[data_center_id]
            acc_curve, roc_curve = _measure_performance_for_each_DC(data_center_name, xlsx_obj)
            acc_curve[data_center_name] = acc_curve.pop('ACC') # rename key
            roc_curve[data_center_name] = roc_curve.pop('ROC') # rename key
            acc_curve[data_center_name]['color'] = curve_color_for_this_DC
            roc_curve[data_center_name]['color'] = curve_color_for_this_DC
            acc_curves.update(acc_curve)
            roc_curves.update(roc_curve)
        acc_fig_save_location = join_path(output_folder, 'acc.pdf')
        roc_fig_save_location = join_path(output_folder, 'roc.pdf')
        multi_curve_plot(acc_curves, acc_fig_save_location, fig_size=(6,6), dpi=150, 
            title='Accuracy of each data center', 
            xlabel='Accept positive class if its probability greater than threshold',
            ylabel='Accuracy',
            xlim=[-0.1,1.1], ylim=[-0.1,1.1])
        multi_curve_plot(roc_curves, roc_fig_save_location, fig_size=(6,6), dpi=150, 
            title='ROC of each data center', 
            xlabel='FPR',
            ylabel='TPR',
            xlim=[-0.1,1.1], ylim=[-0.1,1.1])
        xlsx_obj.save_and_close()


if __name__ == '__main__':
    # unittest
    perf_stat = PerformanceStatistics(['yes', 'no'])
    perf_stat.load_from_xlsx('1.xlsx')
    perf_stat.measure_performance_binary('.')

