#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <filesystem>
#include <chrono>

#include "nmpc_flatness.hpp"
#include "usv_params.h"

namespace fs = std::filesystem;
using std::placeholders::_1;

struct RefPoint {
    double t;
    double x, y, psi;
    double u, v, r;
    double tau_u, tau_r;
    double T1, T2;
    double cmd_l, cmd_r;
};

static double quatToYaw(double qx, double qy, double qz, double qw) {
    double t3 = 2.0 * (qw * qz + qx * qy);
    double t4 = 1.0 - 2.0 * (qy * qy + qz * qz);
    return std::atan2(t3, t4);
}

class Case2MpcNode : public rclcpp::Node {
public:
    Case2MpcNode() : Node("case2_mpc_node") {
        this->declare_parameter<double>("rate", 30.0);
        this->declare_parameter<std::string>("odom_topic",
            "/wamv/sensors/position/ground_truth_odometry");
        this->declare_parameter<std::string>("output_dir",
            "/home/brayan/ros2_ws/src/min-jerk-flatness-planning/Pseudo-Flat_Reconstruction/RealtimeController/output");

        rate_ = this->get_parameter("rate").as_double();
        dt_ = 1.0 / rate_;
        std::string odom_topic = this->get_parameter("odom_topic").as_string();
        output_dir_ = this->get_parameter("output_dir").as_string();
        fs::create_directories(output_dir_);

        pub_lf_ = this->create_publisher<std_msgs::msg::Float64>(
            "/wamv/thrusters/left_front/cmd", 10);
        pub_lr_ = this->create_publisher<std_msgs::msg::Float64>(
            "/wamv/thrusters/left_rear/cmd", 10);
        pub_rf_ = this->create_publisher<std_msgs::msg::Float64>(
            "/wamv/thrusters/right_front/cmd", 10);
        pub_rr_ = this->create_publisher<std_msgs::msg::Float64>(
            "/wamv/thrusters/right_rear/cmd", 10);

        pub_telemetry_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/case2_mpc/telemetry", 10);

        sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic, 10, std::bind(&Case2MpcNode::odomCallback, this, _1));

        rclcpp::QoS latched_qos(1);
        latched_qos.transient_local();
        sub_ref_path_ = this->create_subscription<std_msgs::msg::String>(
            "/case2/reference_csv_path", latched_qos,
            std::bind(&Case2MpcNode::refPathCallback, this, _1));

        nmpc_ = std::make_unique<NmpcFlatness>(dt_);

        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(dt_),
            std::bind(&Case2MpcNode::controlLoop, this));

        RCLCPP_INFO(this->get_logger(),
                    "Case 2 MPC Node initialized (Horizon N=%d, dt=%.4f s, 9-Param Pure Flatness Receding-Horizon NLP MPC). "
                    "Waiting for reference trajectory and odometry...",
                    NmpcFlatness::N, dt_);
    }

    ~Case2MpcNode() override {
        publishThrusters(0.0, 0.0);
        saveResults();
    }

private:
    double rate_, dt_;
    std::string output_dir_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_lf_, pub_lr_, pub_rf_, pub_rr_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_telemetry_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_ref_path_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::unique_ptr<NmpcFlatness> nmpc_;

    bool odom_received_ = false;
    double raw_x_ = 0, raw_y_ = 0, raw_yaw_ = 0;
    double raw_vx_ = 0, raw_vy_ = 0, raw_wz_ = 0;

    bool reference_ready_ = false;
    std::vector<RefPoint> ref_;
    size_t step_idx_ = 0;
    bool started_ = false;
    bool finished_ = false;
    rclcpp::Time start_time_;

    std::array<double, NmpcFlatness::NU> u_prev_{0.0, 0.0};

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        raw_x_ = msg->pose.pose.position.x;
        raw_y_ = msg->pose.pose.position.y;
        double qx = msg->pose.pose.orientation.x;
        double qy = msg->pose.pose.orientation.y;
        double qz = msg->pose.pose.orientation.z;
        double qw = msg->pose.pose.orientation.w;
        raw_yaw_ = quatToYaw(qx, qy, qz, qw);

        raw_vx_ = msg->twist.twist.linear.x;
        raw_vy_ = msg->twist.twist.linear.y;
        raw_wz_ = msg->twist.twist.angular.z;
        odom_received_ = true;
    }

    void refPathCallback(const std_msgs::msg::String::SharedPtr msg) {
        if (reference_ready_) return;
        if (loadReferenceCsv(msg->data)) {
            reference_ready_ = true;
            RCLCPP_INFO(this->get_logger(),
                        "Reference trajectory loaded (%zu samples) from: %s",
                        ref_.size(), msg->data.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(),
                         "Failed to load reference from: %s", msg->data.c_str());
        }
    }

    bool loadReferenceCsv(const std::string &path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;

        std::string header;
        std::getline(f, header);
        std::vector<std::string> cols;
        std::stringstream hs(header);
        std::string colname;
        while (std::getline(hs, colname, ',')) cols.push_back(colname);

        auto idxOf = [&](const std::string &name) -> int {
            for (size_t i = 0; i < cols.size(); i++) if (cols[i] == name) return (int)i;
            return -1;
        };
        int i_t = idxOf("t"), i_x = idxOf("x_ref"), i_y = idxOf("y_ref"),
            i_psi = idxOf("psi_ref"), i_u = idxOf("u_ref"), i_v = idxOf("v_ref"),
            i_r = idxOf("r_ref"), i_tu = idxOf("tau_u_ref"), i_tr = idxOf("tau_r_ref"),
            i_T1 = idxOf("T1_ref"), i_T2 = idxOf("T2_ref"),
            i_cl = idxOf("cmd_left_ref"), i_cr = idxOf("cmd_right_ref");

        if (i_t < 0 || i_x < 0 || i_y < 0 || i_psi < 0 || i_u < 0 || i_v < 0 || i_r < 0) {
            return false;
        }

        ref_.clear();
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::vector<std::string> vals;
            std::stringstream ls(line);
            std::string cell;
            while (std::getline(ls, cell, ',')) vals.push_back(cell);
            if ((int)vals.size() <= i_r) continue;

            RefPoint row{};
            row.t = std::stod(vals[i_t]);
            row.x = std::stod(vals[i_x]);
            row.y = std::stod(vals[i_y]);
            row.psi = std::stod(vals[i_psi]);
            row.u = std::stod(vals[i_u]);
            row.v = std::stod(vals[i_v]);
            row.r = std::stod(vals[i_r]);
            row.tau_u = (i_tu >= 0 && (int)vals.size() > i_tu) ? std::stod(vals[i_tu]) : 0.0;
            row.tau_r = (i_tr >= 0 && (int)vals.size() > i_tr) ? std::stod(vals[i_tr]) : 0.0;
            row.T1 = (i_T1 >= 0 && (int)vals.size() > i_T1) ? std::stod(vals[i_T1]) : 0.0;
            row.T2 = (i_T2 >= 0 && (int)vals.size() > i_T2) ? std::stod(vals[i_T2]) : 0.0;
            row.cmd_l = (i_cl >= 0 && (int)vals.size() > i_cl) ? std::stod(vals[i_cl]) : 0.0;
            row.cmd_r = (i_cr >= 0 && (int)vals.size() > i_cr) ? std::stod(vals[i_cr]) : 0.0;
            ref_.push_back(row);
        }
        return !ref_.empty();
    }

    void publishThrusters(double cmd_l, double cmd_r) {
        std_msgs::msg::Float64 ml, mr;
        ml.data = cmd_l;
        mr.data = cmd_r;
        pub_lf_->publish(ml);
        pub_lr_->publish(ml);
        pub_rf_->publish(mr);
        pub_rr_->publish(mr);
    }

    std::vector<double> log_t_;
    std::vector<double> log_x_real_, log_y_real_, log_psi_real_;
    std::vector<double> log_u_real_, log_v_real_, log_r_real_;
    std::vector<double> log_vx_, log_vy_, log_wz_;
    std::vector<double> log_x_ref_, log_y_ref_, log_psi_ref_;
    std::vector<double> log_u_ref_, log_v_ref_, log_r_ref_;
    std::vector<double> log_tau_u_ref_, log_tau_r_ref_;
    std::vector<double> log_tau_u_app_, log_tau_r_app_;
    std::vector<double> log_cmd_l_, log_cmd_r_;
    std::vector<double> log_solve_ms_;

    void controlLoop() {
        if (finished_) return;
        if (!odom_received_ || !reference_ready_) return;

        if (!started_) {
            started_ = true;
            start_time_ = this->now();
            RCLCPP_INFO(this->get_logger(), "Starting 30Hz closed-loop Case 2 NMPC control execution...");
        }

        if (step_idx_ >= ref_.size()) {
            stopAndFinish();
            return;
        }

        std::array<double, NmpcFlatness::NX> x0{
            raw_x_, -raw_y_, -raw_yaw_, raw_vx_, -raw_vy_, -raw_wz_
        };

        std::vector<std::array<double, 3>> eta_ref, nu_ref;
        std::vector<std::array<double, 2>> tau_ref;
        eta_ref.reserve(NmpcFlatness::N);
        nu_ref.reserve(NmpcFlatness::N);
        tau_ref.reserve(NmpcFlatness::N);
        for (int k = 0; k < NmpcFlatness::N; k++) {
            size_t idx = std::min(step_idx_ + static_cast<size_t>(k), ref_.size() - 1);
            eta_ref.push_back({ref_[idx].x, ref_[idx].y, ref_[idx].psi});
            nu_ref.push_back({ref_[idx].u, ref_[idx].v, ref_[idx].r});
            tau_ref.push_back({ref_[idx].tau_u, ref_[idx].tau_r});
        }

        double solve_ms = 0.0;
        std::array<double, NmpcFlatness::NU> u_opt =
            nmpc_->solve(x0, eta_ref, nu_ref, u_prev_, &solve_ms, tau_ref);
        u_prev_ = u_opt;

        double T1 = u_opt[0], T2 = u_opt[1];
        double cmd_l = cmd_from_thrust_richards(T1);
        double cmd_r = cmd_from_thrust_richards(T2);

        publishThrusters(cmd_l, cmd_r);

        double elapsed = (this->now() - start_time_).seconds();
        const RefPoint &r0 = ref_[step_idx_];

        double T1_act = thrust_from_cmd_richards(cmd_l);
        double T2_act = thrust_from_cmd_richards(cmd_r);
        double tau_u_app = SURGE_GAIN * (T1_act + T2_act);
        double tau_r_app = 2.0 * YAW_ARM * (T1_act - T2_act);

        log_t_.push_back(elapsed);
        log_x_real_.push_back(x0[0]); log_y_real_.push_back(x0[1]); log_psi_real_.push_back(x0[2]);
        log_u_real_.push_back(x0[3]); log_v_real_.push_back(x0[4]); log_r_real_.push_back(x0[5]);
        log_vx_.push_back(raw_vx_); log_vy_.push_back(raw_vy_); log_wz_.push_back(raw_wz_);

        log_x_ref_.push_back(r0.x); log_y_ref_.push_back(r0.y); log_psi_ref_.push_back(r0.psi);
        log_u_ref_.push_back(r0.u); log_v_ref_.push_back(r0.v); log_r_ref_.push_back(r0.r);
        log_tau_u_ref_.push_back(r0.tau_u); log_tau_r_ref_.push_back(r0.tau_r);
        log_tau_u_app_.push_back(tau_u_app); log_tau_r_app_.push_back(tau_r_app);
        log_cmd_l_.push_back(cmd_l); log_cmd_r_.push_back(cmd_r);
        log_solve_ms_.push_back(solve_ms);

        std_msgs::msg::Float64MultiArray telem;
        telem.data = {
            elapsed,
            static_cast<double>(step_idx_),
            static_cast<double>(ref_.size()),
            T1,
            T2,
            cmd_l,
            cmd_r,
            solve_ms
        };
        pub_telemetry_->publish(telem);

        if (step_idx_ % 150 == 0) {
            double err_pos = std::hypot(x0[0] - r0.x, x0[1] - r0.y);
            RCLCPP_INFO(this->get_logger(),
                        "Step %zu/%zu (%.1fs) | Pos:(%.2f,%.2f) Ref:(%.2f,%.2f) | "
                        "ErrPos:%.3fm | solve:%.4fms (N=%d) | T1=%.2fN T2=%.2fN | Cmd:(%.2f,%.2f)",
                        step_idx_, ref_.size(), elapsed, x0[0], x0[1], r0.x, r0.y,
                        err_pos, solve_ms, NmpcFlatness::N, T1, T2, cmd_l, cmd_r);
        }

        step_idx_++;
    }

    void stopAndFinish() {
        if (finished_) return;
        finished_ = true;
        publishThrusters(0.0, 0.0);
        RCLCPP_INFO(this->get_logger(), "Trajectory completed. Thrusters stopped.");

        std_msgs::msg::Float64MultiArray telem;
        telem.data = {
            log_t_.empty() ? 0.0 : log_t_.back(),
            static_cast<double>(ref_.size()),
            static_cast<double>(ref_.size()),
            0.0, 0.0, 0.0, 0.0, 0.0
        };
        pub_telemetry_->publish(telem);

        saveResults();
    }

    void saveResults() {
        if (log_t_.empty()) return;

        fs::path out_path = fs::path(output_dir_);
        fs::create_directories(out_path);
        fs::path csv_path = out_path / "mpc_controller_internal_log.csv";

        std::ofstream f(csv_path);
        if (!f.is_open()) return;

        f << "t,x_real,y_real,psi_real,u_real,v_real,r_real,"
             "vx,vy,wz,x_ref,y_ref,psi_ref,u_ref,v_ref,r_ref,"
             "tau_u_ref,tau_r_ref,tau_u_applied,tau_r_applied,"
             "cmd_left,cmd_right,u_left,u_right,"
             "err_x,err_y,err_pos,err_psi,err_u,err_v,err_r,solve_ms\n";

        double sum_sq = 0.0, max_err = 0.0;
        double sum_solve = 0.0, max_solve = 0.0;
        for (size_t i = 0; i < log_t_.size(); i++) {
            double ex = log_x_real_[i] - log_x_ref_[i];
            double ey = log_y_real_[i] - log_y_ref_[i];
            double epos = std::hypot(ex, ey);
            double epsi = std::atan2(std::sin(log_psi_real_[i] - log_psi_ref_[i]),
                                      std::cos(log_psi_real_[i] - log_psi_ref_[i]));
            double eu = log_u_real_[i] - log_u_ref_[i];
            double ev = log_v_real_[i] - log_v_ref_[i];
            double er = log_r_real_[i] - log_r_ref_[i];
            double sms = (i < log_solve_ms_.size()) ? log_solve_ms_[i] : 0.0;

            sum_sq += epos * epos;
            if (epos > max_err) max_err = epos;
            sum_solve += sms;
            if (sms > max_solve) max_solve = sms;

            f << log_t_[i] << ","
              << log_x_real_[i] << "," << log_y_real_[i] << "," << log_psi_real_[i] << ","
              << log_u_real_[i] << "," << log_v_real_[i] << "," << log_r_real_[i] << ","
              << log_vx_[i] << "," << log_vy_[i] << "," << log_wz_[i] << ","
              << log_x_ref_[i] << "," << log_y_ref_[i] << "," << log_psi_ref_[i] << ","
              << log_u_ref_[i] << "," << log_v_ref_[i] << "," << log_r_ref_[i] << ","
              << log_tau_u_ref_[i] << "," << log_tau_r_ref_[i] << ","
              << log_tau_u_app_[i] << "," << log_tau_r_app_[i] << ","
              << log_cmd_l_[i] << "," << log_cmd_r_[i] << ","
              << log_cmd_l_[i] << "," << log_cmd_r_[i] << ","
              << ex << "," << ey << "," << epos << "," << epsi << ","
              << eu << "," << ev << "," << er << "," << sms << "\n";
        }
        f.close();
        RCLCPP_INFO(this->get_logger(), "Tracking CSV saved to: %s", csv_path.c_str());

        double rmse = std::sqrt(sum_sq / static_cast<double>(log_t_.size()));
        double mean_solve = log_t_.empty() ? 0.0 : (sum_solve / log_t_.size());
        fs::path json_path = out_path / "mpc_controller_metrics.json";
        std::ofstream jf(json_path);
        if (jf.is_open()) {
            jf << "{\n"
               << "    \"case\": \"Case 2 (Pseudo-Flatness 9-Param Pure Flatness Receding-Horizon NLP MPC, N=30)\",\n"
               << "    \"horizon_steps\": " << NmpcFlatness::N << ",\n"
               << "    \"samples_executed\": " << log_t_.size() << ",\n"
               << "    \"duration_s\": " << log_t_.back() << ",\n"
               << "    \"solve_time_ms\": {\n"
               << "        \"mean_solve_ms\": " << mean_solve << ",\n"
               << "        \"max_solve_ms\": " << max_solve << ",\n"
               << "        \"headroom_margin_pct\": " << (1.0 - mean_solve / (dt_ * 1000.0)) * 100.0 << "\n"
               << "    },\n"
               << "    \"tracking_error\": {\n"
               << "        \"rmse_position_m\": " << rmse << ",\n"
               << "        \"max_position_error_m\": " << max_err << "\n"
               << "    }\n"
               << "}\n";
            jf.close();
            RCLCPP_INFO(this->get_logger(), "Metrics saved to: %s (RMSE=%.4fm, Max=%.4fm)",
                        json_path.c_str(), rmse, max_err);
        }
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Case2MpcNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
