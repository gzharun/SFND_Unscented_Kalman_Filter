#include "ukf.h"
#include <cassert>
#include <iostream>

#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd::Zero(5);

  // initial covariance matrix, estimate variance for unmeasured values
  P_ = MatrixXd::Identity(5, 5);
  P_(3,3) = 0.0009;
  P_(4,4) = 0.09;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/4;

  /**
  * DO NOT MODIFY measurement noise values below.
  * These are provided by the sensor manufacturer.
  */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  * End DO NOT MODIFY section for measurement noise values
  */

  /**
  * TODO: Complete the initialization. See ukf.h for other member properties.
  * Hint: one or more values initialized above might be wildly off...
  */
  n_x_ = 5;

  n_aug_ = 7;

  lambda_ = 3 - n_aug_;

  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

  weights_ = VectorXd::Zero(2 * n_aug_ + 1);
  // set weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < weights_.rows(); ++i) {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  eps_laser_ = 0.0;
  eps_radar_ = 0.0;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_) {
    return;
  }

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) {
    return;
  }

  if (is_initialized_) {
    Prediction(meas_package.timestamp_ - time_us_);
  }

  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else {
    assert(false); // error: wrong sensor type
  }
}

void UKF::Prediction(double delta_t) {
  const auto dt = delta_t / 1000000;

  // Generate sigma points
  // Augmented state
  VectorXd x_aug = VectorXd::Zero(7);
  // Augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(7, 7);
  // Sigma point matrix
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);

  x_aug.head(n_x_) = x_;
  x_aug.tail(2) << 0, 0;

  // Augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2, 2) << std_a_*std_a_, 0, 0, std_yawdd_*std_yawdd_;

  // Square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  // Generate augmented sigma points
  Xsig_aug.col(0) = x_aug;
  MatrixXd C1 = A * std::sqrt(lambda_ + n_aug_);
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i+1) = x_aug + C1.col(i);
    Xsig_aug.col(i+1 + n_aug_) = x_aug - C1.col(i);
  }

  // Predict sigma points
  for (int i = 0; i < Xsig_aug.cols(); ++i) {
    float px = Xsig_aug(0, i);
    float py = Xsig_aug(1, i);
    float v = Xsig_aug(2, i);
    float psi = Xsig_aug(3, i);
    float psi_d = Xsig_aug(4, i);
    float nua = Xsig_aug(5, i);
    float nup = Xsig_aug(6, i);

    float dt2 = dt * dt / 2;
    if (std::abs(psi_d) < 0.001) {
      px += v * cos(psi) * dt;
      py += v * sin(psi) * dt;
    } else {
      px += v * (sin(psi + psi_d * dt) - sin(psi)) / psi_d;
      py += v * (-cos(psi + psi_d * dt) + cos(psi)) / psi_d;
    }
    px += dt2 * cos(psi) * nua;
    py += dt2 * sin(psi) * nua;
    v += dt * nua;
    psi += psi_d * dt + dt2 * nup;
    psi_d += dt * nup;

    Xsig_pred_.col(i) << px, py, v, psi, psi_d;
  }

  // Prediction
  x_.fill(0.);
  for (int i = 0; i < Xsig_pred_.rows(); ++i) {
    x_(i) = Xsig_pred_.row(i) * weights_;
  }

  P_.fill(0.);
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    VectorXd x_pred_centered = Xsig_pred_.col(i) - x_;
    while (x_pred_centered(3) > M_PI) x_pred_centered(3) -= 2*M_PI;
    while (x_pred_centered(3) <-M_PI) x_pred_centered(3) += 2*M_PI;
    P_ += weights_(i) * x_pred_centered * x_pred_centered.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    return;
  }

  // Measurements dimention
  int n_z = 2;

  // Sigma points into measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < Zsig.cols(); ++i) {
    float px = Xsig_pred_(0, i);
    float py = Xsig_pred_(1, i);

    Zsig.col(i) << px, py;
  }

  // Mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);
  for (int i = 0; i < Zsig.rows(); ++i) {
    z_pred(i) = Zsig.row(i) * weights_;
  }

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  for (int i = 0; i < Zsig.cols(); ++i) {
    VectorXd z_pred_centered = Zsig.col(i) - z_pred;
    while (z_pred_centered(1) > M_PI) z_pred_centered(1) -= 2*M_PI;
    while (z_pred_centered(1) <-M_PI) z_pred_centered(1) += 2*M_PI;
    S += weights_(i) * z_pred_centered * z_pred_centered.transpose();
  }

  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;
  S += R;

  // Cross correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    VectorXd x_centered = Xsig_pred_.col(i) - x_;
    while (x_centered(3) > M_PI) x_centered(3) -= 2*M_PI;
    while (x_centered(3) <-M_PI) x_centered(3) += 2*M_PI;

    VectorXd z_centered = Zsig.col(i) - z_pred;
    while (z_centered(1) > M_PI) z_centered(1) -= 2*M_PI;
    while (z_centered(1) <-M_PI) z_centered(1) += 2*M_PI;

    Tc += weights_(i) * x_centered * z_centered.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = MatrixXd::Zero(n_x_, n_z);
  K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd z_centered = meas_package.raw_measurements_ - z_pred;
  x_ += K * z_centered;
  P_ -= K * S * K.transpose();

  eps_laser_ = z_centered.transpose() * S.inverse() * z_centered;

  time_us_ = meas_package.timestamp_;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    x_ << meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]),
          meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]),
          meas_package.raw_measurements_[2],
          meas_package.raw_measurements_[1], 0;
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    return;
  }

  // Measurements dimention
  int n_z = 3;

  // Sigma points in measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < Zsig.cols(); ++i) {
    const float p_x = Xsig_pred_(0,i);
    const float p_y = Xsig_pred_(1,i);
    const float v  = Xsig_pred_(2,i);
    const float phi = Xsig_pred_(3,i);

    const float vx = cos(phi) * v;
    const float vy = sin(phi) * v;

    // Measurement model
    const float r = sqrt(p_x*p_x + p_y*p_y);
    const float psi = atan2(p_y, p_x);
    const float rd = (p_x * vx + p_y * vy) / r;
    Zsig.col(i) << r, psi, rd;
  }

  // Mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);
  for (int i = 0; i < Zsig.rows(); ++i) {
    z_pred(i) = Zsig.row(i) * weights_;
  }

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  for (int i = 0; i < Zsig.cols(); ++i) {
    VectorXd z_pred_centered = Zsig.col(i) - z_pred;
    while (z_pred_centered(1) > M_PI) z_pred_centered(1) -= 2*M_PI;
    while (z_pred_centered(1) <-M_PI) z_pred_centered(1) += 2*M_PI;
    S += weights_(i) * z_pred_centered * z_pred_centered.transpose();
  }

  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R << std_radr_ * std_radr_, 0, 0,
       0, std_radphi_ * std_radphi_, 0,
       0, 0, std_radrd_ * std_radrd_;
  S += R;

  // Cross correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    VectorXd x_centered = Xsig_pred_.col(i) - x_;
    while (x_centered(3) > M_PI) x_centered(3) -= 2*M_PI;
    while (x_centered(3) <-M_PI) x_centered(3) += 2*M_PI;

    VectorXd z_centered = Zsig.col(i) - z_pred;
    while (z_centered(1) > M_PI) z_centered(1) -= 2*M_PI;
    while (z_centered(1) <-M_PI) z_centered(1) += 2*M_PI;

    Tc += weights_(i) * x_centered * z_centered.transpose();
  }

  // Kalman gain K calculation
  MatrixXd K = MatrixXd::Zero(n_x_, n_z);
  K = Tc * S.inverse();

  // State mean and covariance matrix update
  VectorXd z_centered = meas_package.raw_measurements_ - z_pred;
  while (z_centered(1)> M_PI) z_centered(1) -= 2.*M_PI;
  while (z_centered(1)<-M_PI) z_centered(1) += 2.*M_PI;
  x_ += K * z_centered;
  P_ -= K * S * K.transpose();

  eps_radar_ = z_centered.transpose() * S.inverse() * z_centered;

  time_us_ = meas_package.timestamp_;
}