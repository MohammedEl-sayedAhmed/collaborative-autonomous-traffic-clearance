// The lockstep plant plugin (M5): commands in and state out through synchronous services,
// so a referee tick is exact and repeatable. See LockstepPlant.cc.
#ifndef CAATC_LOCKSTEP_PLANT_HH_
#define CAATC_LOCKSTEP_PLANT_HH_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/double_v.pb.h>
#include <gz/msgs/empty.pb.h>
#include <gz/msgs/stringmsg_v.pb.h>
#include <gz/sim/Entity.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>

namespace caatc
{
  struct CarHandles
  {
    std::string name;
    gz::sim::Entity model{gz::sim::kNullEntity};
    gz::sim::Entity chassis{gz::sim::kNullEntity};
    gz::sim::Entity chassisCollision{gz::sim::kNullEntity};
    std::vector<gz::sim::Entity> steerJoints;   // left, right hinge
    std::vector<gz::sim::Entity> wheelJoints;   // rear left, rear right
    double targetSteer{0.0};
    double targetSpeed{0.0};
    double speedCmd{0.0};                       // the rate-limited wheel speed actually commanded
    double lastSteer{0.0};                      // the hinge angle read last step (for the report)
    bool contact{false};                        // touched anything since the last command
  };

  /// One system per world. It drives every listed car from the last command it was given
  /// (a steering angle and a speed per car, like f1tenth_gym's action) with the 2D model's
  /// actuator limits, and reports every car's pose, speed, steering angle and contacts.
  class LockstepPlant : public gz::sim::System,
                        public gz::sim::ISystemConfigure,
                        public gz::sim::ISystemPreUpdate,
                        public gz::sim::ISystemPostUpdate
  {
  public:
    void Configure(const gz::sim::Entity &_entity, const std::shared_ptr<const sdf::Element> &_sdf,
                   gz::sim::EntityComponentManager &_ecm, gz::sim::EventManager &_eventMgr) override;
    void PreUpdate(const gz::sim::UpdateInfo &_info, gz::sim::EntityComponentManager &_ecm) override;
    void PostUpdate(const gz::sim::UpdateInfo &_info, const gz::sim::EntityComponentManager &_ecm) override;

  private:
    bool OnCommand(const gz::msgs::Double_V &_req, gz::msgs::Boolean &_rep);
    bool OnState(const gz::msgs::Empty &_req, gz::msgs::Double_V &_rep);
    bool OnContacts(const gz::msgs::Empty &_req, gz::msgs::StringMsg_V &_rep);

    gz::transport::Node node;
    std::mutex mutex;
    std::vector<CarHandles> cars;
    std::vector<std::string> carNames;
    double wheelRadius{0.05};
    double steerMax{0.4189};
    double steerRateMax{3.2};
    double steerGain{30.0};
    double accelMax{9.51};
    double speedMax{20.0};
    double wheelBase{0.3302};
    double trackWidth{0.2};
    bool resolved{false};
    // the report, refreshed every PostUpdate
    uint64_t iterations{0};
    double simTime{0.0};
    std::vector<double> report;
    uint64_t commandsReceived{0};
    std::vector<std::string> contactLog;        // "sim_time car collision_a | collision_b", first 500
  };
}
#endif
