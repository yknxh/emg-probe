import unittest

import numpy as np

from emgprobe.filters import EMGFilter, filter_settings


class EMGFilterTests(unittest.TestCase):
    def test_removes_off_center_mains_and_harmonics_far_above_emg(self):
        # Recorded sessions: mains at 59.95/60.05 Hz, ~2e5 counts hum vs ~1e2 counts resting EMG.
        rate, t = 2000, np.arange(20*2000)/2000
        emg = np.random.default_rng(3).normal(0, 100, (len(t), 8))
        hum = 2e5*np.sin(2*np.pi*59.95*t)+2e4*np.sin(2*np.pi*3*59.95*t+1)+5e3*np.sin(2*np.pi*5*59.95*t+2)
        clean = EMGFilter(rate).process(emg)[1]
        noisy = EMGFilter(rate).process(emg+hum[:, None])[1]
        tail = slice(5*rate, None)
        np.testing.assert_allclose(np.nanmedian(noisy[tail], 0), np.nanmedian(clean[tail], 0), rtol=.15)

    def test_settings_describe_v3_cascade(self):
        s = filter_settings(2000)
        self.assertEqual((s['version'], s['notch_hz'], s['notch_bandwidth_hz'], s['notch_passes']), (3, [60, 120, 180, 240, 300, 360, 420], 4, 3))
        self.assertEqual(len(EMGFilter(2000).sos), 4+7*3)

    def test_keeps_emg_band_between_harmonics(self):
        rate, t = 2000, np.arange(10*2000)/2000
        tone = 1000*np.sin(2*np.pi*np.array([90, 150, 330])[None, :]*t[:, None])
        rms = EMGFilter(rate, channels=3).process(tone)[1][5*rate:]
        np.testing.assert_allclose(np.nanmedian(rms, 0), 1000/np.sqrt(2), rtol=.1)


if __name__ == '__main__':
    unittest.main()
