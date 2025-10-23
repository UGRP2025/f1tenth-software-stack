// Copyright (C) 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023 Tino Kluge
// Copyright (C) 2021, 2022, 2023 Steffen A. Mork
//
// This file is part of the "spline" library.
// It is distributed under the MIT License.
// A copy of the license is available in the LICENSE file.

#ifndef SPLINE_H
#define SPLINE_H

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#ifdef HAVE_SSTREAM
#include <sstream>
#endif

// A simple class for cubic spline interpolation.
// It is not optimized for performance and does not support arbitrary
// functions. So if you have a large number of points, or need to
// interpolate functions that are not mostly smooth, then you should
// probably use a different library.
//
// See: https://github.com/s-mork/spline
class Spline
{
public:
	enum class bd_type
	{
		first_deriv = 1,
		second_deriv = 2
	};

private:
	// x, y, a, b, c, d=y
	std::vector<double> m_x, m_y, m_a, m_b, m_c;
	bd_type m_left_type, m_right_type;
	double  m_left_value, m_right_value;
	bool    m_force_linear_extrapolation;

public:
	// set default boundary condition to be zero curvature at both ends
	Spline() :
		m_left_type(bd_type::second_deriv), m_right_type(bd_type::second_deriv),
		m_left_value(0.0), m_right_value(0.0),
		m_force_linear_extrapolation(false)
	{
	}

	Spline(
		const std::vector<double> & x,
		const std::vector<double> & y,
		bd_type left_type = bd_type::second_deriv, double left_value = 0.0,
		bd_type right_type = bd_type::second_deriv, double right_value = 0.0,
		bool force_linear_extrapolation = false) :
		m_left_type(left_type), m_right_type(right_type),
		m_left_value(left_value), m_right_value(right_value),
		m_force_linear_extrapolation(force_linear_extrapolation)
	{
		set_points(x, y);
	}

	void set_boundary(
		bd_type left_type, double left_value,
		bd_type right_type, double right_value)
	{
		m_left_type = left_type;
		m_right_type = right_type;
		m_left_value = left_value;
		m_right_value = right_value;
	}

	// x must be strictly increasing
	void set_points(
		const std::vector<double> & x,
		const std::vector<double> & y)
	{
		if (x.size() != y.size())
		{
#ifdef HAVE_SSTREAM
			std::stringstream ss;
			ss << "Spline: x.size() (" << x.size() << ") != y.size() (" << y.size() << ")";
			throw std::invalid_argument(ss.str());
#else
			throw std::invalid_argument("Spline: x.size() != y.size()");
#endif
		}
		if (x.size() < 2)
		{
			throw std::invalid_argument("Spline: at least 2 points are required");
		}
		for (size_t i = 0; i < x.size() - 1; i++)
		{
			if (x[i] >= x[i + 1])
			{
#ifdef HAVE_SSTREAM
				std::stringstream ss;
				ss << "Spline: x must be strictly increasing, but x[" << i << "]=" << x[i] << " and x[" << i + 1 << "]=" << x[i + 1];
				throw std::invalid_argument(ss.str());
#else
				throw std::invalid_argument("Spline: x must be strictly increasing");
#endif
			}
		}

		m_x = x;
		m_y = y;
		const size_t n = x.size();

		// setting up the matrix and right hand side of the equation system
		// for the parameters b[]
		std::vector<double> h(n - 1);
		for (size_t i = 0; i < n - 1; i++)
		{
			h[i] = m_x[i + 1] - m_x[i];
		}

		// Tridiagonal matrix
		std::vector<double> A(n - 2), B(n - 2), C(n - 2);
		std::vector<double> r(n - 2);
		for (size_t i = 0; i < n - 2; i++)
		{
			A[i] = h[i];
			B[i] = 2.0 * (h[i] + h[i + 1]);
			C[i] = h[i + 1];
			r[i] = 3.0 * ((m_y[i + 2] - m_y[i + 1]) / h[i + 1] - (m_y[i + 1] - m_y[i]) / h[i]);
		}

		// boundary conditions
		if (m_left_type == bd_type::second_deriv)
		{
			B[0] -= h[0] * h[0] / (2.0 * h[0]);
			r[0] -= h[0] * m_left_value;
		}
		else if (m_left_type == bd_type::first_deriv)
		{
			B[0] -= h[0] * 0.5;
			r[0] -= 3.0 * ((m_y[1] - m_y[0]) / h[0] - m_left_value);
		}

		if (m_right_type == bd_type::second_deriv)
		{
			B[n - 3] -= h[n - 2] * h[n - 2] / (2.0 * h[n - 2]);
			r[n - 3] -= h[n - 2] * m_right_value;
		}
		else if (m_right_type == bd_type::first_deriv)
		{
			B[n - 3] -= h[n - 2] * 0.5;
			r[n - 3] -= 3.0 * (m_right_value - (m_y[n - 1] - m_y[n - 2]) / h[n - 2]);
		}

		// solve the tridiagonal system
		for (size_t i = 1; i < n - 2; i++)
		{
			const double m = A[i] / B[i - 1];
			B[i] -= m * C[i - 1];
			r[i] -= m * r[i - 1];
		}
		// back substitution
		m_b.resize(n);
		m_b[n - 2] = r[n - 3] / B[n - 3];
		for (size_t i = n - 4; i > 0; i--)
		{
			m_b[i + 1] = (r[i] - C[i] * m_b[i + 2]) / B[i];
		}
		m_b[1] = (r[0] - C[0] * m_b[2]) / B[0];

		// boundary conditions
		if (m_left_type == bd_type::second_deriv)
		{
			m_b[0] = m_b[1] - h[0] * m_left_value;
		}
		else if (m_left_type == bd_type::first_deriv)
		{
			m_b[0] = 2.0 * (m_y[1] - m_y[0]) / h[0] - m_b[1] - 3.0 * m_left_value;
		}

		if (m_right_type == bd_type::second_deriv)
		{
			m_b[n - 1] = m_b[n - 2] + h[n - 2] * m_right_value;
		}
		else if (m_right_type == bd_type::first_deriv)
		{
			m_b[n - 1] = 2.0 * (m_y[n - 1] - m_y[n - 2]) / h[n - 2] - m_b[n - 2] + 3.0 * m_right_value;
		}

		// compute a and c
		m_a.resize(n);
		m_c.resize(n);
		for (size_t i = 0; i < n - 1; i++)
		{
			m_a[i] = (m_b[i + 1] - m_b[i]) / (3.0 * h[i]);
			m_c[i] = (m_y[i + 1] - m_y[i]) / h[i] - h[i] * (m_b[i + 1] + 2.0 * m_b[i]) / 3.0;
		}
	}

	// returns the interpolated y-value for a given x-value.
	// extrapolation is done linearly, if m_force_linear_extrapolation is true
	// or with the cubic polynomial of the last segment, if false.
	double operator()(double x) const
	{
		const size_t n = m_x.size();
		// find the closest point to the left
		size_t idx = 0;
		if (x > m_x[0])
		{
			auto it = std::upper_bound(m_x.begin(), m_x.end(), x);
			idx = std::max(size_t(0), size_t(std::distance(m_x.begin(), it) - 1));
		}

		// extrapolate to the left
		if (x < m_x[0])
		{
			return m_force_linear_extrapolation ?
				linear_extrapolate(m_x[0], m_y[0], m_c[0], x) :
				cubic_extrapolate(0, x);
		}
		// extrapolate to the right
		else if (x > m_x[n - 1])
		{
			return m_force_linear_extrapolation ?
				linear_extrapolate(m_x[n - 1], m_y[n - 1], m_c[n - 2] + (m_b[n - 2] + m_a[n - 2] * (m_x[n - 1] - m_x[n - 2])) * (m_x[n - 1] - m_x[n - 2]), x) :
				cubic_extrapolate(n - 2, x);
		}
		// interpolate
		else
		{
			return cubic_extrapolate(idx, x);
		}
	}

	// returns the first derivative for a given x-value.
	double deriv(double x) const
	{
		const size_t n = m_x.size();
		// find the closest point to the left
		size_t idx = 0;
		if (x > m_x[0])
		{
			auto it = std::upper_bound(m_x.begin(), m_x.end(), x);
			idx = std::max(size_t(0), size_t(std::distance(m_x.begin(), it) - 1));
		}

		const double h = x - m_x[idx];
		return m_c[idx] + (2.0 * m_b[idx] + 3.0 * m_a[idx] * h) * h;
	}

	// returns the second derivative for a given x-value.
	double deriv2(double x) const
	{
		const size_t n = m_x.size();
		// find the closest point to the left
		size_t idx = 0;
		if (x > m_x[0])
		{
			auto it = std::upper_bound(m_x.begin(), m_x.end(), x);
			idx = std::max(size_t(0), size_t(std::distance(m_x.begin(), it) - 1));
		}

		const double h = x - m_x[idx];
		return 2.0 * m_b[idx] + 6.0 * m_a[idx] * h;
	}

private:
	double cubic_extrapolate(size_t idx, double x) const
	{
		const double h = x - m_x[idx];
		return m_y[idx] + (m_c[idx] + (m_b[idx] + m_a[idx] * h) * h) * h;
	}

	double linear_extrapolate(double x1, double y1, double m1, double x) const
	{
		return y1 + m1 * (x - x1);
	}
};

#endif // SPLINE_H
