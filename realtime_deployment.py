import numpy as np
import absl.logging
import pyqtgraph as pg
import tensorflow as tf
from tensorflow import keras
absl.logging.set_verbosity(absl.logging.ERROR)

from acconeer.exptool.pg_process import PGProcess
from acconeer.exptool.a121._core.peripherals import load_record

import acconeer.exptool as et
from acconeer.exptool import a121
tf.compat.v1.disable_eager_execution()

from utils import estimate_distance, calculate_variance_at_fixed_distance


def main():

    args = a121.ExampleArgumentParser().parse_args()
    et.utils.config_logging(args)

    client = a121.Client.open()

    session_config = load_record('data/carpet_1.h5').session_config
    sensor_config = session_config.sensor_config

    print(session_config.sensor_id)
    metadata = client.setup_session(session_config)

    processor = Processor(sensor_config)
    pg_updater = PGUpdater(sensor_config, metadata)
    pg_process = PGProcess(pg_updater)
    pg_process.start()

    client.start_session()
    interrupt_handler = et.utils.ExampleInterruptHandler()

    while not interrupt_handler.got_signal:
        data = client.get_next()
        plot_Data = processor.process(data)
        pg_process.put_data(plot_Data)

    print("\nDisconnecting...")
    client.close()


class Processor:
    def __init__(self, sensor_config):
        self.sensor_config = sensor_config
        self.model = keras.models.load_model('model')
        self.classes = ['carpet', 'floor']

    def process(self, data):
        frame = data.frame

        # extract peak location
        distance_to_floor = estimate_distance(frame, self.sensor_config)
        if len(distance_to_floor) == 0:
            return {"y_pred": np.nan}

        feature_distance_to_floor = distance_to_floor[0]

        # extract variance data
        feature_variance = calculate_variance_at_fixed_distance(frame)

        y_pred = self.model.predict(np.expand_dims([feature_distance_to_floor, feature_variance], axis=0))[0]

        return {"y_pred": y_pred, "feature_distance_to_floor": feature_distance_to_floor, "feature_variance": feature_variance}

class PGUpdater:
    def __init__(self, sensor_config, metadata):
        history_horizon = 200
        self.class_history = [np.nan] * history_horizon
        self.distance_history = [np.nan] * history_horizon
        self.variance_history = [np.nan] * history_horizon

        self.fifo = [0]*5

    def setup(self, win):
        pen = et.utils.pg_pen_cycler(0)
        brush = et.utils.pg_brush_cycler(0)
        symbol_kw = dict(symbol="h", symbolSize=1, symbolBrush=brush, symbolPen="k")
        feat_kw = dict(pen=pen, **symbol_kw)

        self.class_plot = win.addPlot(row=0, col=0, clospan=2)
        self.class_plot.setMenuEnabled(False)
        self.class_plot.showGrid(x=True, y=True)
        self.class_plot.addLegend()
        self.class_plot.setLabel("left", "Probability of being carpet")
        self.class_plot.addItem(pg.PlotDataItem())
        self.class_curve = self.class_plot.plot(**feat_kw)
        self.class_plot.setYRange(-0.1, 1.1)

        self.distance_plot = win.addPlot(row=1, col=0, clospan=2)
        self.distance_plot.setMenuEnabled(False)
        self.distance_plot.showGrid(x=True, y=True)
        self.distance_plot.addLegend()
        self.distance_plot.setLabel("left", "Distance to floor (m)")
        self.distance_plot.addItem(pg.PlotDataItem())
        self.distance_curve = self.distance_plot.plot(**feat_kw)

        self.smooth_lim_distance = et.utils.SmoothLimits()

        self.variance_plot = win.addPlot(row=2, col=0, clospan=2)
        self.variance_plot.setMenuEnabled(False)
        self.variance_plot.showGrid(x=True, y=True)
        self.variance_plot.addLegend()
        self.variance_plot.setLabel("left", "Variance", size=20)
        self.variance_plot.setLabel("bottom", "Distance(m)")
        self.variance_plot.addItem(pg.PlotDataItem())
        self.variance_curve = self.variance_plot.plot(**feat_kw)

        self.smooth_lim_variance = et.utils.SmoothLimits()

    def update(self, d):
        if not np.any(np.isnan(d["y_pred"])):

            self.fifo.pop(0)
            self.fifo.append(d["y_pred"][0])

            self.class_history.pop(0)
            self.class_history.append(np.mean(self.fifo))

            self.distance_history.pop(0)
            self.distance_history.append(d["feature_distance_to_floor"])
            distance_lims = self.smooth_lim_distance.update(self.distance_history)

            self.variance_history.pop(0)
            self.variance_history.append(d["feature_variance"])
            variance_lims = self.smooth_lim_variance.update(self.variance_history)

            self.class_curve.setData(self.class_history)
            self.distance_curve.setData(self.distance_history)
            self.variance_curve.setData(self.variance_history)

            self.distance_plot.setYRange(distance_lims[0], distance_lims[1])
            self.variance_plot.setYRange(variance_lims[0], variance_lims[1])


if __name__ == "__main__":
    main()
