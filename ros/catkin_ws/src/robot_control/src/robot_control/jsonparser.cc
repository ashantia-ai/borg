/*
 * jsonparser.cc
 *
 *  Created on: 15 jan. 2014
 *      Author: Amir Shantia
 */
#include "robotdriver.ih"

bool RobotDriver::jsonParser(string &type, bool &clockWise, double &amount)
{
	borg_pioneer::MemoryReadSrv srv;
	srv.request.function = "get_last_observation";
	srv.request.timestamp = ros::Time::now();
	srv.request.name = "navigation_command";
	srv.request.params = "";

	// Make the call
	if (not client_reader.call(srv))
	{

		ROS_WARN_ONCE("No connection\n");
		return false;
	}
	// Read message
	string msg = srv.response.json;

	// Let's parse it
	Json::Value root;
	Json::Reader reader;
	bool parsedSuccess = reader.parse(msg, root, false);

	if (not parsedSuccess)
	{
		// Report failures and their locations
		// in the document.
		ROS_ERROR_STREAM("Failed to parse JSON\n" << reader.getFormatedErrorMessages() << '\n');
		return false;
	}

	//Extracting Parts
	Json::Value const time = root["time"];
	double cmd_time;
	string cmd;

	if (not time.isNull())
	{
		cmd_time = time.asDouble();

		//Only continue if the command is new
		if (cmd_time <= last_command)
			return false;
		ROS_DEBUG_STREAM("command_time " << cmd_time);
		ROS_DEBUG_STREAM("last command_time " << last_command);
		last_command = cmd_time;
	}

	Json::Value const command = root["type"];

	if (not command.isNull())
	{
		type = command.asString();
	}

	Json::Value const cw = root["clockWise"];
	if (not cw.isNull())
	{
		clockWise = cw.asBool();
	}

	Json::Value const value = root["amount"];
	if (not value.isNull())
	{
		amount = value.asDouble();
	}

	if (not time.isNull() and not command.isNull() and not cw.isNull() and not value.isNull())
		// If we want to print JSON is as easy as doing:
		return true;
	else
		return false;

	return true;
}


