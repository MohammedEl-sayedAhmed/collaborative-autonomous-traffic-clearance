// The lockstep plant plugin (M5, ADR 0013).
//
// Why it exists: driving the cars over a topic and then asking the world to step is a race.
// The command message and the step request travel on different sockets, so in some runs the
// command lands one physics step late, and two identical runs end a few micrometres apart.
// Here the commands arrive through a SERVICE: the call returns only after this plugin has
// stored them, so the step request that follows always sees them. The state goes out through
// a second service that reports the last completed step, exact, no stamps to match.
//
// The actuator model is the 2D model's: a steering angle target reached at most at
// steerRateMax (3.2 rad/s), a speed target reached at most at accelMax (9.51 m/s^2), both
// clipped to the same limits f1tenth_gym uses. Steering uses Ackermann geometry on the two
// hinges; the rear wheels are velocity-driven.
//
// Services (world-level, one plugin per world):
//   /caatc/lockstep/command   gz.msgs.Double_V [steer0, speed0, steer1, speed1, ...] -> Boolean
//   /caatc/lockstep/state     gz.msgs.Empty -> Double_V [iterations, sim_time, n,
//                             then per car: x, y, yaw, v, delta, contact-since-last-command]
//   /caatc/lockstep/contacts  gz.msgs.Empty -> StringMsg_V, every contact pair heard (first 500)
#include "LockstepPlant.hh"

#include <algorithm>
#include <cmath>

#include <gz/common/Console.hh>
#include <gz/math/Pose3.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/ContactSensorData.hh>
#include <gz/sim/components/Joint.hh>
#include <gz/sim/components/JointPosition.hh>
#include <gz/sim/components/JointVelocity.hh>
#include <gz/sim/components/JointVelocityCmd.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Pose.hh>

using namespace caatc;
using namespace gz;
using namespace gz::sim;

namespace
{
  double clampd(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }

  double param(const std::shared_ptr<const sdf::Element> &_sdf, const std::string &_name, double _default)
  {
    return _sdf->HasElement(_name) ? _sdf->Get<double>(_name) : _default;
  }
}

void LockstepPlant::Configure(const Entity &, const std::shared_ptr<const sdf::Element> &_sdf,
                              EntityComponentManager &, EventManager &)
{
  auto sdf = _sdf->Clone();
  if (sdf->HasElement("model"))
  {
    for (auto e = sdf->GetElement("model"); e; e = e->GetNextElement("model"))
      this->carNames.push_back(e->Get<std::string>());
  }
  this->wheelRadius = param(_sdf, "wheel_radius", this->wheelRadius);
  this->steerMax = param(_sdf, "steer_max", this->steerMax);
  this->steerRateMax = param(_sdf, "steer_rate_max", this->steerRateMax);
  this->steerGain = param(_sdf, "steer_gain", this->steerGain);
  this->accelMax = param(_sdf, "accel_max", this->accelMax);
  this->speedMax = param(_sdf, "speed_max", this->speedMax);
  this->wheelBase = param(_sdf, "wheel_base", this->wheelBase);
  this->trackWidth = param(_sdf, "track_width", this->trackWidth);

  this->node.Advertise("/caatc/lockstep/command", &LockstepPlant::OnCommand, this);
  this->node.Advertise("/caatc/lockstep/state", &LockstepPlant::OnState, this);
  this->node.Advertise("/caatc/lockstep/contacts", &LockstepPlant::OnContacts, this);
  gzmsg << "[caatc lockstep] " << this->carNames.size() << " cars, services /caatc/lockstep/{command,state}" << std::endl;
}

bool LockstepPlant::OnCommand(const msgs::Double_V &_req, msgs::Boolean &_rep)
{
  std::lock_guard<std::mutex> lock(this->mutex);
  const int n = static_cast<int>(this->cars.size());
  if (_req.data_size() != 2 * n)
  {
    gzerr << "[caatc lockstep] command with " << _req.data_size() << " numbers for " << n << " cars" << std::endl;
    _rep.set_data(false);
    return true;
  }
  for (int i = 0; i < n; ++i)
  {
    this->cars[i].targetSteer = clampd(_req.data(2 * i), -this->steerMax, this->steerMax);
    this->cars[i].targetSpeed = clampd(_req.data(2 * i + 1), -this->speedMax, this->speedMax);
    this->cars[i].contact = false;                 // "touched anything since the last command"
  }
  this->commandsReceived++;
  _rep.set_data(true);
  return true;
}

bool LockstepPlant::OnState(const msgs::Empty &, msgs::Double_V &_rep)
{
  std::lock_guard<std::mutex> lock(this->mutex);
  _rep.clear_data();
  _rep.add_data(static_cast<double>(this->iterations));
  _rep.add_data(this->simTime);
  _rep.add_data(static_cast<double>(this->cars.size()));
  for (double v : this->report)
    _rep.add_data(v);
  return true;
}

bool LockstepPlant::OnContacts(const msgs::Empty &, msgs::StringMsg_V &_rep)
{
  std::lock_guard<std::mutex> lock(this->mutex);
  _rep.clear_data();
  for (const auto &line : this->contactLog)
    _rep.add_data(line);
  return true;
}

void LockstepPlant::PreUpdate(const UpdateInfo &_info, EntityComponentManager &_ecm)
{
  if (!this->resolved)
  {
    // find the cars once every model is spawned
    std::vector<CarHandles> found;
    for (const auto &name : this->carNames)
    {
      Entity m = _ecm.EntityByComponents(components::Name(name), components::Model());
      if (m == kNullEntity)
        return;                                    // not all there yet: try next step
      Model model(m);
      CarHandles h;
      h.name = name;
      h.model = m;
      h.chassis = model.LinkByName(_ecm, "chassis");
      for (auto j : {"front_left_steer_joint", "front_right_steer_joint"})
        h.steerJoints.push_back(model.JointByName(_ecm, j));
      for (auto j : {"rear_left_wheel_joint", "rear_right_wheel_joint"})
        h.wheelJoints.push_back(model.JointByName(_ecm, j));
      for (auto j : h.steerJoints) { if (!_ecm.Component<components::JointPosition>(j)) _ecm.CreateComponent(j, components::JointPosition()); }
      for (auto j : h.wheelJoints) { if (!_ecm.Component<components::JointVelocity>(j)) _ecm.CreateComponent(j, components::JointVelocity()); }
      // the chassis collision named "c": the contact system fills its ContactSensorData
      _ecm.Each<components::Name, components::ParentEntity>(
        [&](const Entity &_e, const components::Name *_n, const components::ParentEntity *_p) -> bool
        {
          if (_p->Data() == h.chassis && _n->Data() == "c") h.chassisCollision = _e;
          return true;
        });
      if (h.chassisCollision != kNullEntity && !_ecm.Component<components::ContactSensorData>(h.chassisCollision))
        _ecm.CreateComponent(h.chassisCollision, components::ContactSensorData());
      found.push_back(h);
    }
    std::lock_guard<std::mutex> lock(this->mutex);
    this->cars = found;
    this->report.assign(6 * this->cars.size(), 0.0);
    this->resolved = true;
    gzmsg << "[caatc lockstep] resolved " << this->cars.size() << " cars" << std::endl;
  }
  if (_info.paused)
    return;
  const double dt = std::chrono::duration<double>(_info.dt).count();
  std::lock_guard<std::mutex> lock(this->mutex);
  for (auto &car : this->cars)
  {
    // speed: a rate-limited target, like the 2D model's acceleration limit
    const double dv = clampd(car.targetSpeed - car.speedCmd, -this->accelMax * dt, this->accelMax * dt);
    car.speedCmd += dv;
    const double wheelRate = car.speedCmd / this->wheelRadius;
    for (auto j : car.wheelJoints)
    {
      auto cmd = _ecm.Component<components::JointVelocityCmd>(j);
      if (!cmd) _ecm.CreateComponent(j, components::JointVelocityCmd({wheelRate}));
      else cmd->Data() = {wheelRate};
    }
    // steering: Ackermann geometry on the hinges, each moved toward its target at most at steerRateMax
    const double t = std::tan(car.targetSteer);
    const double inner = std::atan(this->wheelBase * t / (this->wheelBase - 0.5 * this->trackWidth * t));
    const double outer = std::atan(this->wheelBase * t / (this->wheelBase + 0.5 * this->trackWidth * t));
    const double targets[2] = {car.targetSteer >= 0 ? inner : outer, car.targetSteer >= 0 ? outer : inner};   // left, right
    for (size_t k = 0; k < car.steerJoints.size(); ++k)
    {
      auto pos = _ecm.Component<components::JointPosition>(car.steerJoints[k]);
      const double cur = (pos && !pos->Data().empty()) ? pos->Data()[0] : 0.0;
      const double rate = clampd(this->steerGain * (targets[k] - cur), -this->steerRateMax, this->steerRateMax);
      auto cmd = _ecm.Component<components::JointVelocityCmd>(car.steerJoints[k]);
      if (!cmd) _ecm.CreateComponent(car.steerJoints[k], components::JointVelocityCmd({rate}));
      else cmd->Data() = {rate};
    }
  }
}

void LockstepPlant::PostUpdate(const UpdateInfo &_info, const EntityComponentManager &_ecm)
{
  if (!this->resolved || _info.paused)
    return;
  std::lock_guard<std::mutex> lock(this->mutex);
  this->iterations = _info.iterations;
  this->simTime = std::chrono::duration<double>(_info.simTime).count();
  for (size_t i = 0; i < this->cars.size(); ++i)
  {
    auto &car = this->cars[i];
    const math::Pose3d pose = worldPose(car.model, _ecm);
    double rate = 0.0; int nr = 0;
    for (auto j : car.wheelJoints)
    {
      auto v = _ecm.Component<components::JointVelocity>(j);
      if (v && !v->Data().empty()) { rate += v->Data()[0]; ++nr; }
    }
    double steer = 0.0; int ns = 0;
    for (auto j : car.steerJoints)
    {
      auto p = _ecm.Component<components::JointPosition>(j);
      if (p && !p->Data().empty()) { steer += p->Data()[0]; ++ns; }
    }
    bool contact = false;
    if (car.chassisCollision != kNullEntity)
    {
      auto c = _ecm.Component<components::ContactSensorData>(car.chassisCollision);
      contact = c && c->Data().contact_size() > 0;
      if (contact && this->contactLog.size() < 500)
      {
        for (int k = 0; k < c->Data().contact_size(); ++k)
          this->contactLog.push_back(std::to_string(this->simTime) + " " + car.name + " " +
                                     c->Data().contact(k).collision1().name() + " | " + c->Data().contact(k).collision2().name());
      }
    }
    car.contact = car.contact || contact;
    this->report[6 * i + 0] = pose.Pos().X();
    this->report[6 * i + 1] = pose.Pos().Y();
    this->report[6 * i + 2] = pose.Rot().Yaw();
    this->report[6 * i + 3] = nr ? (rate / nr) * this->wheelRadius : 0.0;
    this->report[6 * i + 4] = ns ? steer / ns : 0.0;
    this->report[6 * i + 5] = car.contact ? 1.0 : 0.0;
  }
}

GZ_ADD_PLUGIN(caatc::LockstepPlant, gz::sim::System, LockstepPlant::ISystemConfigure,
              LockstepPlant::ISystemPreUpdate, LockstepPlant::ISystemPostUpdate)
GZ_ADD_PLUGIN_ALIAS(caatc::LockstepPlant, "caatc::LockstepPlant")
