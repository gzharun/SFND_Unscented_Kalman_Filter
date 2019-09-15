#include "ukf.h"
#include <cassert>

#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI / 8;
  
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

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  weights = VectorXd(2 * n_aug_ + 1);
  // set weights
  weights(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < weights.rows(); ++i) {
    weights(i) = 0.5 / (lambda_ + n_aug_);
  }
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
    // Generate sigma points
    // Augmented state
    VectorXd x_aug = VectorXd(7);
    // Augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);
    // Sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

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

        float dt2 = delta_t * delta_t / 2;
        if (std::abs(psi_d) < 0.001) {
            px += v * cos(psi) * delta_t;
            py += v * sin(psi) * delta_t;
        } else {
            px += v * (sin(psi + psi_d * delta_t) - sin(psi)) / psi_d;
            py += v * (-cos(psi + psi_d * delta_t) + cos(psi)) / psi_d;
            psi += psi_d * delta_t;
        }
        px += dt2 * cos(psi) * nua;
        py += dt2 * sin(psi) * nua;
        v += delta_t * nua;
        psi += dt2 * nup;
        psi_d += delta_t * nup;

        Xsig_pred_.col(i) << px, py, v, psi, psi_d;
    }

    // Prediction
    for (int i = 0; i < Xsig_pred_.rows(); ++i) {
        x_(i) = Xsig_pred_.row(i) * weights;
    }

    for (int i = 0; i < Xsig_pred_.cols(); ++i) {
        VectorXd x_pred_centered = Xsig_pred_.col(i) - x_;
        while (x_pred_centered(3) > M_PI) x_pred_centered(3) -= 2*M_PI;
        while (x_pred_centered(3) <-M_PI) x_pred_centered(3) += 2*M_PI;
        P_ += weights(i) * x_pred_centered * x_pred_centered.transpose();
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

    // Sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    // Mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);

    // Measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);

    // transform sigma points into measurement space
    for (int i = 0; i < Zsig.cols(); ++i) {
        float px = Xsig_pred_(0, i);
        float py = Xsig_pred_(1, i);

        Zsig.col(i) << px, px;
    }

    // Mean predicted measurement
    for (int i = 0; i < Zsig.rows(); ++i) {
        z_pred(i) = Zsig.row(i) * weights;
    }

    // Measurement covariance matrix S
    for (int i = 0; i < Zsig.cols(); ++i) {
        VectorXd z_pred_centered(n_z);
        z_pred_centered = Zsig.col(i) - z_pred;

        S += weights(i) * z_pred_centered * z_pred_centered.transpose();
    }

    MatrixXd R(n_z, n_z);
    R << std_laspx_ * std_laspx_, 0,
            0, std_laspy_ * std_laspy_;
    S += R;

    // Cross correlation matrix
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    for (int i = 0; i < Xsig_pred_.cols(); ++i) {
        VectorXd x_centered = Xsig_pred_.col(i) - x_;
        VectorXd z_centered = Zsig.col(i) - z_pred;

        Tc += weights(i) * x_centered * z_centered.transpose();
    }

    // calculate Kalman gain K;
    MatrixXd K(n_x_, n_z);
    K = Tc * S.inverse();

    // update state mean and covariance matrix
    x_ += K * (meas_package.raw_measurements_ - z_pred);
    P_ -= K * S * K.transpose();
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
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    // Mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);

    // Measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);

    // transform sigma points into measurement space
    for (int i = 0; i < Zsig.cols(); ++i) {
        float r = std::sqrt(Xsig_pred_(0, i) * Xsig_pred_(0, i) +
                            Xsig_pred_(1, i) * Xsig_pred_(1, i));
        float psi = std::atan(Xsig_pred_(1, i) / Xsig_pred_(0, i));
        float r_d = Xsig_pred_(2, i) * (Xsig_pred_(0, i) * cos(Xsig_pred_(3, i)) +
                    Xsig_pred_(1, i) * sin(Xsig_pred_(3, i))) / r;

        Zsig.col(i) << r, psi, r_d;
    }

    // Mean predicted measurement
    for (int i = 0; i < Zsig.rows(); ++i) {
        z_pred(i) = Zsig.row(i) * weights;
    }

    // Measurement covariance matrix S
    for (int i = 0; i < Zsig.cols(); ++i) {
        VectorXd z_pred_centered(n_z);
        z_pred_centered = Zsig.col(i) - z_pred;

        S += weights(i) * z_pred_centered * z_pred_centered.transpose();
    }

    MatrixXd R(n_z, n_z);
    R << std_radr_ * std_radr_, 0, 0,
         0, std_radphi_ * std_radphi_, 0,
         0, 0, std_radrd_ * std_radrd_;
    S += R;

    // Cross correlation matrix
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    for (int i = 0; i < Xsig_pred_.cols(); ++i) {
        VectorXd x_centered = Xsig_pred_.col(i) - x_;
        VectorXd z_centered = Zsig.col(i) - z_pred;

        Tc += weights(i) * x_centered * z_centered.transpose();
    }

    // calculate Kalman gain K;
    MatrixXd K(n_x_, n_z);
    K = Tc * S.inverse();

    // update state mean and covariance matrix
    x_ += K * (meas_package.raw_measurements_ - z_pred);
    P_ -= K * S * K.transpose();
}