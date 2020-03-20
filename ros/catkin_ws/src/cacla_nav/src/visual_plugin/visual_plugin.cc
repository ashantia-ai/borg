#include "visual_plugin.h"
#include <iostream>
namespace gazebo
{
  namespace rendering
  {

    ////////////////////////////////////////////////////////////////////////////////
    // Constructor
    GoalVisualPlugin::GoalVisualPlugin(): 
      line(NULL)
    {

    }

    ////////////////////////////////////////////////////////////////////////////////
    // Destructor
    GoalVisualPlugin::~GoalVisualPlugin()
    {
      // Finalize the visualizer
      this->rosnode_->shutdown();
      delete this->rosnode_;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Load the plugin
    void GoalVisualPlugin::Load( VisualPtr _parent, sdf::ElementPtr _sdf )
    {
      std::cout << "Loading Goal visualization plugin\n";
      this->visual_ = _parent;

      this->visual_namespace_ = "visual/";

      // start ros node
      if (!ros::isInitialized())
      {
        int argc = 0;
        char** argv = NULL;
        ros::init(argc,argv,"gazebo_visual",ros::init_options::NoSigintHandler|ros::init_options::AnonymousName);
      }

      this->rosnode_ = new ros::NodeHandle(this->visual_namespace_);
      this->force_sub_ = this->rosnode_->subscribe("/some_force", 1, &GoalVisualPlugin::VisualizeForceOnLink, this);

      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->update_connection_ = event::Events::ConnectRender(
          boost::bind(&GoalVisualPlugin::UpdateChild, this));
    }

    //////////////////////////////////////////////////////////////////////////////////
    // Update the visualizer
    void GoalVisualPlugin::UpdateChild()
    {
      ros::spinOnce();
    }

    //////////////////////////////////////////////////////////////////////////////////
    // VisualizeForceOnLink
    void GoalVisualPlugin::VisualizeForceOnLink(const geometry_msgs::PointConstPtr &force_msg)
    {
      this->line = this->visual_->CreateDynamicLine(RENDERING_LINE_STRIP);
/*
      //TODO: Get the current link position
      link_pose = CurrentLinkPose();
      //TODO: Get the current end position
      endpoint = CalculateEndpointOfForceVector(link_pose, force_msg);

      // Add two points to a connecting line strip from link_pose to endpoint
      this->line->AddPoint(
        math::Vector3(
          link_pose.position.x,
          link_pose.position.y,
          link_pose.position.z
          )
        );
      this->line->AddPoint(math::Vector3(endpoint.x, endpoint.y, endpoint.z));
*/
      //this->line->AddPoint(math::Vector3(3.66, -0.59, 0.0));
      //this->line->AddPoint(math::Vector3(3.66, -0.59, 1.0));
      this->line->AddPoint(math::Vector3(4.0, -0.5, 0.0));
      this->line->AddPoint(math::Vector3(4.0, -0.5, 1.0));
      // set the Material of the line, in this case to purple
      this->line->setMaterial("Gazebo/Purple");
      this->line->setVisibilityFlags(GZ_VISIBILITY_GUI);
      this->visual_->SetVisible(true);
    }

    // Register this plugin within the simulator
    GZ_REGISTER_VISUAL_PLUGIN(GoalVisualPlugin)
  }
}
