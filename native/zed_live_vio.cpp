// basalt_zed_live_vio: live Basalt VIO fed over stdin, poses out on stdout.
//
// stdin  (little-endian records):
//   u8 type=1 (IMU)   i64 t_ns  f64 gx gy gz ax ay az
//   u8 type=2 (FRAME) i64 t_ns  u16 w  u16 h  u8 left[w*h]  u8 right[w*h]
// stdout (one line per VIO state):
//   P t_ns px py pz qx qy qz qw vx vy vz bgx bgy bgz bax bay baz
// usage: basalt_zed_live_vio <calib.json> <config.json>
//
// Built inside the Basalt source tree (third_party/basalt), linked against libbasalt.

#include <cstdio>
#include <fstream>
#include <iostream>
#include <thread>

#include <tbb/concurrent_queue.h>

#include <basalt/calibration/calibration.hpp>
#include <basalt/optical_flow/optical_flow.h>
#include <basalt/serialization/headers_serialization.h>
#include <basalt/utils/vio_config.h>
#include <basalt/vi_estimator/vio_estimator.h>

static bool read_exact(void* dst, size_t n) { return std::fread(dst, 1, n, stdin) == n; }

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <calib.json> <config.json>\n";
    return 2;
  }
  basalt::Calibration<double> calib;
  {
    std::ifstream is(argv[1], std::ios::binary);
    if (!is) { std::cerr << "cannot open calib " << argv[1] << "\n"; return 1; }
    cereal::JSONInputArchive ar(is);
    ar(calib);
  }
  basalt::VioConfig config;
  config.load(argv[2]);

  auto opt_flow = basalt::OpticalFlowFactory::getOpticalFlow(config, calib);
  auto vio = basalt::VioEstimatorFactory::getVioEstimator(config, calib, basalt::constants::g, true, true);
  vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  // Default capacity is 300 (~0.4 s at 800 Hz). Live frames lag the IMU by ~70 ms plus a startup sync-lock period;
  // with a full IMU queue the stdin reader blocks before any frame arrives -> deadlock. Give it ~20 s of headroom.
  vio->imu_data_queue.set_capacity(16000);
  opt_flow->output_queue = &vio->vision_data_queue;
  tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr> out_state_queue;
  vio->out_state_queue = &out_state_queue;

  std::thread printer([&] {
    basalt::PoseVelBiasState<double>::Ptr s;
    char line[512];
    while (true) {
      out_state_queue.pop(s);
      if (!s) break;
      const auto& T = s->T_w_i;
      const auto t = T.translation();
      const auto q = T.unit_quaternion();
      std::snprintf(line, sizeof line,
                    "P %lld %.6f %.6f %.6f %.7f %.7f %.7f %.7f %.5f %.5f %.5f %.6f %.6f %.6f %.5f %.5f %.5f\n",
                    (long long)s->t_ns, t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w(), s->vel_w_i.x(),
                    s->vel_w_i.y(), s->vel_w_i.z(), s->bias_gyro.x(), s->bias_gyro.y(), s->bias_gyro.z(),
                    s->bias_accel.x(), s->bias_accel.y(), s->bias_accel.z());
      std::fputs(line, stdout);
      std::fflush(stdout);
    }
  });

  std::cerr << "zed_live_vio: ready (" << calib.intrinsics.size() << " cams)\n";
  int64_t n_imu = 0, n_frames = 0;
  while (true) {
    uint8_t type;
    int64_t t_ns;
    if (!read_exact(&type, 1) || !read_exact(&t_ns, 8)) break;
    if (type == 1) {
      double v[6];
      if (!read_exact(v, sizeof v)) break;
      basalt::ImuData<double>::Ptr d(new basalt::ImuData<double>);
      d->t_ns = t_ns;
      d->gyro = Eigen::Vector3d(v[0], v[1], v[2]);
      d->accel = Eigen::Vector3d(v[3], v[4], v[5]);
      vio->imu_data_queue.push(d);
      n_imu++;
    } else if (type == 2) {
      uint16_t wh[2];
      if (!read_exact(wh, sizeof wh)) break;
      const size_t n = size_t(wh[0]) * wh[1];
      std::vector<uint8_t> buf(2 * n);
      if (!read_exact(buf.data(), buf.size())) break;
      basalt::OpticalFlowInput::Ptr in(new basalt::OpticalFlowInput);
      in->t_ns = t_ns;
      in->img_data.resize(2);
      for (int c = 0; c < 2; c++) {
        in->img_data[c].img.reset(new basalt::ManagedImage<uint16_t>(wh[0], wh[1]));
        uint16_t* out = in->img_data[c].img->ptr;
        const uint8_t* src = buf.data() + c * n;
        for (size_t j = 0; j < n; j++) out[j] = uint16_t(src[j]) << 8;
      }
      opt_flow->input_queue.push(in);
      n_frames++;
    } else {
      std::cerr << "zed_live_vio: bad record type " << int(type) << "\n";
      break;
    }
  }
  std::cerr << "zed_live_vio: input closed after " << n_frames << " frames, " << n_imu << " imu\n";
  opt_flow->input_queue.push(nullptr);
  vio->imu_data_queue.push(nullptr);
  vio->maybe_join();
  out_state_queue.push(nullptr);
  printer.join();
  return 0;
}
