/**
 * particle_filter.cpp
 *
 * Created on: July 7th, 2020
 * Author: Carissa Chan
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>

#include "particle_filter.h"

using namespace std;
using std::string;
using std::vector;

static std::default_random_engine gen;

static int NUM_PARTICLES = 100;

/* Set the number of particles.
 * Initialize all particles to first position (based on estimates of
 * x, y, theta and their uncertainties from GPS) and all weights to 1.
 * Add random Gaussian noise to each particle.
 */
void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = NUM_PARTICLES;  // TODO: Set the number of particles
  particles.resize(num_particles);
  
  //Set std deviations for x, y, and theta 
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2]; 
  
  
  //Create a nomal distribution for y and theta using GPS measurment as initial estimate 
  /*std::normal_distribution<> generates random numbers according to the Normal (or Gaussian) random number distribution
  */
  std::normal_distribution<double> N_dist_x(x, std_x);
  std::normal_distribution<double> N_dist_y(y, std_y);
  std::normal_distribution<double> N_dist_theta(theta, std_theta);
  
  /* For each particle, set its x, y, and orientation to a random value chosen within the range of the 
   * estimated location. The standard deviation is based on an accuracy uncertainty of the sensor  
   * responsible for measuring that axis value. Assign all particle weights to 1 as they all initialy have an 
   * equal chance of being the correct pseudo location without prior history or data collection.
   */
  for(auto& p: particles){
      p.x = N_dist_x(gen);
      p.y = N_dist_y(gen);
      p.theta = N_dist_theta(gen);
      p.weight = 1.0;
    
      particles.push_back(p);
    
  }  
	is_initialized = true;

}

/* Add measurements to each particle and add random Gaussian noise.
 */
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  std::default_random_engine gen;

  // generate random Gaussian noise
  std::normal_distribution<double> N_x(0, std_pos[0]);
  std::normal_distribution<double> N_y(0, std_pos[1]);
  std::normal_distribution<double> N_theta(0, std_pos[2]);

  for(auto& p: particles) {

    // add measurements to each particle
    if( fabs(yaw_rate) < 0.0001){  // constant velocity
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);

    } else{
      p.x += velocity / yaw_rate * ( sin( p.theta + yaw_rate*delta_t ) - sin(p.theta) );
      p.y += velocity / yaw_rate * ( cos( p.theta ) - cos( p.theta + yaw_rate*delta_t ) );
      p.theta += yaw_rate * delta_t;
    }

    // predicted particles with added sensor noise
    p.x += N_x(gen);
    p.y += N_y(gen);
    p.theta += N_theta(gen);
  }

}

/* Find the predicted measurement that is closest to each observed measurement and assign the
 * observed measurement to this particular landmark, with exhausted search (may be replaced with KD-tree approaches)
 */
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

 /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(auto& obs: observations) {
    double min_dist = std::numeric_limits<double>::max();

    for(const auto& pred: predicted){
      double distance = dist(obs.x, obs.y, pred.x, pred.y);
      if( min_dist > distance) {
        min_dist = distance;
        obs.id = pred.id;
      }
    }
  }
}
/**
* 4: Update the weights of each particle using a multi-variate Gaussian distribution,
*    https://en.wikipedia.org/wiki/Multivariate_normal_distribution
*/

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, const Map &map_landmarks) {
  for(auto& p: particles){
    p.weight = 1.0;

    // create a vector to store landmark locations predicted within sensor range
    vector<LandmarkObs> predictions;
    
    //Collect landmarks within sensor range and find the distance between the particle and those predicted landmarks
    for(const auto& lm: map_landmarks.landmark_list){
      double distance = dist(p.x, p.y, lm.x_f, lm.y_f);
      if( distance < sensor_range) { // if the landmark is within the sensor range, save it to predictions
        predictions.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
      }
    }
 
      /* create another vector to store the observations that have been transformed to map coordinates via discrete
       * transformation, namely rotation and translation.
       */
      vector<LandmarkObs> observations_map;
      // convert observations from vehicle to map coorindates 
      double cos_theta = cos(p.theta);
      double sin_theta = sin(p.theta);
    
	 //create a temp struct to hold the transformed Landmark observation before inserting into vector
      for(const auto& obs: observations) {
        LandmarkObs tmp;
        tmp.x = obs.x * cos_theta - obs.y * sin_theta + p.x;
        tmp.y = obs.x * sin_theta + obs.y * cos_theta + p.y;
        //tmp.id = obs.id; // maybe an unnecessary step, since the each obersation will get the id from dataAssociation step.
        observations_map.push_back(tmp);
      }
    
    	//Invoke dataAssociation fxn to find the respective landmark index for each observation
      dataAssociation(predictions, observations_map);
    
      /* After particles have been drawn, update the weights of each particle. This determines
      * the probability that the particle should be drawn again when resampling.
      */
      for (const auto& obs_mapped: observations_map){
      	Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_mapped.id - 1);
        double obs_weight = calc_weight(landmark, obs_mapped, std_landmark);	
        p.weight *= obs_weight;
      }
    
      weights.push_back(p.weight);
  }
  
}

/* Resample particles with replacement with probability proportional to their weight.
 * Reference 1: std::discrete_distribution, see the link below with an example
 * http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
 */
void ParticleFilter::resample() {
  
  vector<Particle> new_particles;
//   new_particles.resize(num_particles);
  
  // Create weight vector.
  vector<double> weights;
  
   // Retreive and store all of the current weights
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }
  
   // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());
  
  // uniform random distribution [0.0, max_weight)
  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  // Creating distributions of particle indices 
  uniform_int_distribution<int> distInt(0, num_particles - 1);

  // Randomly chooses an index from the distribution created. 
  auto index = distInt(gen);

  double beta = 0.0;

  // draw an index at random according to a uniform distribution 
  for(int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while(beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
  
  // clear the weight vector for the next round
  weights.clear();
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                       const std::vector<double>& sense_x, 
                       const std::vector<double>& sense_y) {
	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
