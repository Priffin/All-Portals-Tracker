# All Portals Tracker

All Portals Tracker created using Flask. 

- Reads record file to automatically switch to next stronghold
- Integrates with NinjabrainBot API to automatically input coords for the first eight strongholds
- Shows a graph of visited strongholds
- Direction + Distance to stronghold updates with F3+C
- Run the application and go to [localhost:5123](http://localhost:5123) to see the tracker.
- You can also view the tracker on a mobile device if you use a port forwarding service like bore: e.g. **bore local 5123 --to bore.pub**

TODO:
- Use IGT from record file to calculate splits and estimate pace
- Visual timer until pathfinding complete
- Better graph 
- Colour coded angle delta 
- Better OBS browser source option
- Better mobile layout 

Acknowledgments:

@thetalkingmime @shnenanigans @luvvlyjude - Original APGui Tracker

@Lincoln-LM - Optimized pathfinding 
