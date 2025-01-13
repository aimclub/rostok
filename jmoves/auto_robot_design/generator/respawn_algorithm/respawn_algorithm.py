import networkx as nx
import numpy as np
from typing import Tuple, List


from auto_robot_design.description.kinematics import JointPoint
from auto_robot_design.description.builder import add_branch
from auto_robot_design.description.utils import draw_joint_point


class Generator:
    def __init__(
        self,
        num_main_links: Tuple[float] = (2, 3, 4),
        length_link_bounds: Tuple[float] = (0.1, 0.9),
        length_main_branch_bounds: Tuple[float] = (0.6, 1.0),
        dof_mechanism: float = 2,
        length_factor: float = 1.2,
        width_factor: float = 0.5,
    ) -> None:
        self.num_main_link = num_main_links
        self.length_link_bounds = length_link_bounds
        self.length_main_branch_bounds = length_main_branch_bounds
        self.dof_mechanism = dof_mechanism
        self.length_factor = length_factor
        self.width_factor = width_factor

    def create_main_branch(
        self, angle_bounds: Tuple[float] = (-np.pi * 2 / 3, -np.pi / 3)
    ):
        # sample the number of links in the main branch
        num_links = np.random.choice(self.num_main_link)
        # sample lengths of links and normalize them
        length_main_branch = np.random.uniform(*self.length_main_branch_bounds)
        lengths = np.random.uniform(*self.length_link_bounds, size=num_links)
        lengths /= np.sum(lengths)
        lengths *= length_main_branch
        ground_joint = JointPoint(
            r=np.zeros(3),
            w=np.array([0, 1, 0]),
            attach_ground=True,
            active=True,
            name="G0main",
        )

        body_counter = 1
        joint_counter = 1
        main_branch = [ground_joint]
        for i in range(num_links - 1):
            sampled_angle = np.random.uniform(*angle_bounds)
            x = lengths[i] * np.cos(sampled_angle)
            y = 0
            z = lengths[i] * np.sin(sampled_angle)
            new_coordinates = main_branch[-1].r + np.array([x, y, z])
            main_branch.append(
                JointPoint(r=new_coordinates, w=np.array([0, 1, 0]), name=f"J{i}m")
            )
            body_counter += 1
            joint_counter += 1
        sampled_angle = np.random.uniform(*angle_bounds)
        x = lengths[-1] * np.cos(sampled_angle)
        y = 0
        z = lengths[-1] * np.sin(sampled_angle)
        new_coordinates = main_branch[-1].r + np.array([x, y, z])
        main_branch.append(
            JointPoint(
                r=new_coordinates,
                w=np.array([0, 1, 0]),
                name="Jee",
                attach_endeffector=True,
            )
        )
        body_counter += 1
        return main_branch, 3 * (body_counter - 1 - joint_counter) + 1 * joint_counter

    def find_connect_point(
        self,
        first_point: np.array,
        second_point: np.array,
    ) -> np.array:
        center = (first_point + second_point) / 2
        half_length = np.linalg.norm(second_point - first_point) / 2
        direction = second_point - center
        ort_direction = np.cross(direction, np.array([0, 1, 0]))
        sample_length_shift = np.random.uniform(-self.length_factor, self.length_factor)
        sample_width_shift = np.random.uniform(-self.width_factor, self.width_factor)
        total_shift = (
            direction * sample_length_shift
            + ort_direction
            / np.linalg.norm(ort_direction)
            * (self.length_factor - abs(sample_length_shift))
            * sample_width_shift
            * half_length
        )
        pos = center + total_shift

        return pos

    def sample_secondary_branch(
        self,
        graph,
        main_branch,
        length_range: Tuple[float] = (0.3, 0.6),
        dof_reduction: int = 0,
        branch_id=0,
    ):
        local_joints = 0
        local_bodies = 0
        length_constrains = length_range
        # chose first body to connect joint. Secondary branch always starts from some body.
        # each branch can be attached to a body from second branch only once
        idx_set = set(range(len(main_branch)))
        idx_sample = np.random.choice(list(idx_set))
        idx_set.difference_update(
            {idx_sample}
        )  # remove sampled element from index list
        secondary_branch = []  # initialize secondary branch
        if idx_sample != 0:
            # get positions of first and second joints
            first_joint = main_branch[idx_sample - 1].r
            second_joint = main_branch[idx_sample].r
            pos = self.find_connect_point(first_joint, second_joint)
            new_joint = JointPoint(r=pos, w=np.array([0, 1, 0]), name=f"J0_{branch_id}")
            local_joints += 1
            secondary_branch = [
                [main_branch[idx_sample - 1], main_branch[idx_sample]],
                new_joint,
            ]
        else:
            # idx == 0 means attached to ground
            pos = self.find_connect_point(np.array([0, 0, 0]), np.array([0, 0, 0.2]))
            new_joint = JointPoint(
                r=pos, w=np.array([0, 1, 0]), attach_ground=True, name=f"J0_{branch_id}"
            )
            local_joints += 1
            secondary_branch.append(new_joint)

        # usually number link in the branch is 4
        max_link_num = 4
        num_link = np.random.choice(range(1, max_link_num))
        # idx from the main branch that shows the attachment to a body from main branch
        new_joint.attached = idx_sample
        current_joint = new_joint
        attach = False
        i = 0
        while not attach > 0:
            i += 1
            # randomly choose if the new joint is to be attached to main branch
            if num_link == i:
                attach = True
            if attach:
                # the joint is to be attached to a body from the main branch
                if current_joint.attached != -1:
                    # the current is attached to body and the new to be attached, hence the bodies shouldn't be adjacent
                    attached_idx = current_joint.attached
                    # removes the links between adjacent bodies
                    new_set = idx_set.difference(
                        set([attached_idx - 1, attached_idx, attached_idx + 1])
                    )
                    if len(new_set) == 0:
                        return False  # there is no way to connect the branch
                    idx_sample = np.random.choice(list(new_set))
                else:
                    idx_sample = np.random.choice(list(idx_set))

                idx_set.difference_update({idx_sample})  # drop used index from the set
                # find a point to attach
                if idx_sample != 0:
                    # get positions of first and second joints
                    first_joint = main_branch[idx_sample - 1].r
                    second_joint = main_branch[idx_sample].r
                    pos = self.find_connect_point(first_joint, second_joint)

                    new_joint = JointPoint(
                        r=pos,
                        w=np.array([0, 1, 0]),
                        attach_endeffector=False,
                        name=f"J{i}_{branch_id}",
                    )
                    local_joints += 1
                    local_bodies += 1
                    secondary_branch += [
                        new_joint,
                        [main_branch[idx_sample - 1], main_branch[idx_sample]],
                    ]
                else:
                    pos = self.find_connect_point(
                        np.array([0, 0, 0]), np.array([0, 0, 0.2])
                    )
                    new_joint = JointPoint(
                        r=pos,
                        w=np.array([0, 1, 0]),
                        active=False,
                        attach_ground=True,
                        name=f"J{i}_{branch_id}",
                    )
                    secondary_branch.append(new_joint)
                    local_joints += 1
                    local_bodies += 1
                new_joint.attached = idx_sample
            else:
                # we don't need a point to be attached on bodies, hence we only sample a random point, but it must be below z = 0
                new_pos = self.find_connect_point(
                    np.array([0, 0, 0]), np.array([0, 0, 0.2])
                )

                new_joint = JointPoint(
                    r=new_pos,
                    w=np.array([0, 1, 0]),
                    active=False,
                    attach_endeffector=False,
                    name=f"J{i}_{branch_id}",
                )
                new_joint.attached = -1
                secondary_branch.append(new_joint)
                local_joints += 1
                local_bodies += 1
                current_joint = new_joint

        # if the initial secondary branch reduces dof too hard the building is failed
        delta_dof = 3 * local_bodies - 2 * local_joints
        if delta_dof < dof_reduction:
            return False
        # get joint from secondary branch
        triangle_list = [x for x in secondary_branch if type(x) is JointPoint]
        triangle_idx = set(range(1, len(triangle_list)))
        if delta_dof - dof_reduction > min(len(triangle_idx), len(idx_set)):
            return False

        add_branch(graph, secondary_branch)
        j = 0
        while delta_dof > dof_reduction:
            j += 1
            i += 1
            # if len(triangle_idx)==0 or len(idx_set)==0: return False
            triangle_sample = np.random.choice(list(triangle_idx))
            triangle_idx.difference_update({triangle_sample})
            idx_sample = np.random.choice(list(idx_set))
            idx_set.difference_update({idx_sample})
            first_joint_triangle = triangle_list[triangle_sample - 1].r
            second_joint_triangle = triangle_list[triangle_sample].r
            new_pos = self.find_connect_point(
                first_joint_triangle, second_joint_triangle
            )
            new_joint_triangle = JointPoint(
                r=new_pos,
                w=np.array([0, 1, 0]),
                active=False,
                attach_endeffector=False,
                name=f"J{i}_{j}_{branch_id}",
            )
            i += 1
            if idx_sample != 0:
                first_joint = main_branch[idx_sample - 1].r
                second_joint = main_branch[idx_sample].r
                new_pos = self.find_connect_point(first_joint, second_joint)
                new_joint = JointPoint(
                    r=new_pos,
                    w=np.array([0, 1, 0]),
                    active=False,
                    attach_endeffector=False,
                    name=f"J{i}_{j}_{branch_id}",
                )
                new_branch = [
                    [
                        triangle_list[triangle_sample - 1],
                        triangle_list[triangle_sample],
                    ],
                    new_joint_triangle,
                    new_joint,
                    [main_branch[idx_sample - 1], main_branch[idx_sample]],
                ]
            else:
                pos = self.find_connect_point(
                    np.array([0, 0, 0]), np.array([0, 0, 0.2])
                )
                new_joint = JointPoint(
                    r=pos,
                    w=np.array([0, 1, 0]),
                    active=False,
                    attach_ground=True,
                    name=f"J{i}_{branch_id}",
                )
                new_branch = [
                    [
                        triangle_list[triangle_sample - 1],
                        triangle_list[triangle_sample],
                    ],
                    new_joint_triangle,
                    new_joint,
                ]
            local_bodies += 1
            local_joints += 2
            add_branch(graph, new_branch)
            delta_dof = 3 * local_bodies - 2 * local_joints

        return graph


def generate_graph():
    graph = nx.Graph()
    generator = Generator()
    main_branch, dof = generator.create_main_branch()
    add_branch(graph, main_branch)
    # dof = body_counter * 3 - 2 * joint_counter
    # # print(dof)
    zero_reduction = True
    b_idx = 0
    while dof > 2 or zero_reduction:
        if zero_reduction:
            sample_dof_reduction = np.random.randint(0, dof - 1)
            zero_reduction = False
        else:
            sample_dof_reduction = np.random.randint(1, dof - 1)
        i = 0
        while not generator.sample_secondary_branch(
            graph, main_branch, branch_id=b_idx, dof_reduction=-sample_dof_reduction
        ):
            if i > 50:
                return False
            i += 1
        dof -= sample_dof_reduction
        b_idx += 1

    return graph


class RespawnAlgorithm(Generator):
    def __init__(
        self,
        groups_libary,
        num_main_links: Tuple[float] = (2, 3, 4),
        length_link_bounds: Tuple[float] = (0.1, 0.9),
        length_main_branch_bounds: Tuple[float] = (0.6, 1),
        dof_mechanism: float = 2,
        length_factor: float = 1.2,
        width_factor: float = 0.5,
    ) -> None:
        super().__init__(
            num_main_links,
            length_link_bounds,
            length_main_branch_bounds,
            dof_mechanism,
            length_factor,
            width_factor,
        )
        self.groups_libary = groups_libary
        
    
    def sample_secondary_branch(
        self,
        graph,
        main_branch,
        length_range: Tuple[float] = (0.3, 0.6),
        dof_reduction: int = 0,
        branch_id=0,
    ):
        local_joints = 0
        local_bodies = 0
        length_constrains = length_range
        # chose first body to connect joint. Secondary branch always starts from some body.
        # each branch can be attached to a body from second branch only once
        idx_set = set(range(len(main_branch)))
        idx_sample = np.random.choice(list(idx_set))
        idx_set.difference_update(
            {idx_sample}
        )  # remove sampled element from index list
        
        all_groups = {g.dof:g.get_all_groups() for g in self.groups_libary}
        dof_group = np.random.choice(all_groups.keys())
        group, id_outs = np.random.choice(all_groups[dof_group])

        idx_set = set(range(len(main_branch)))
        for id in id_outs:
            idx_sample = np.random.choice(list(idx_set))
            idx_set.difference_update(
                    {idx_sample}
                )  # remove sampled element from index list
            
            if idx_sample != 0:
                # get positions of first and second joints
                first_joint = main_branch[idx_sample - 1].r
                second_joint = main_branch[idx_sample].r
                group[id[0]][id[1]] = [main_branch[idx_sample-1],main_branch[idx_sample]]
            else:
                # idx == 0 means attached to ground
                pos = self.find_connect_point(np.array([0, 0, 0]), np.array([0, 0, 0.2]))
                new_joint = JointPoint(
                    r=pos, w=np.array([0, 1, 0]), attach_ground=True, name=f"J0_{branch_id}"
                )
                group[id[0]][id[1]] = new_joint
        
        # idx from the main branch that shows the attachment to a body from main branch
        new_joint.attached = idx_sample
        current_joint = new_joint
        attach = False
        i = 0

        # if the initial secondary branch reduces dof too hard the building is failed
        delta_dof = 3 * local_bodies - 2 * local_joints
        if delta_dof < dof_reduction:
            return False
        # get joint from secondary branch
        triangle_list = [x for x in secondary_branch if type(x) is JointPoint]
        triangle_idx = set(range(1, len(triangle_list)))
        if delta_dof - dof_reduction > min(len(triangle_idx), len(idx_set)):
            return False

        add_branch(graph, secondary_branch)
        j = 0

        return graph

if __name__ == "__main__":
    for i in range(1000):
        body_counter = 0
        joint_counter = 0
        graph = generate_graph()

    if graph:
        draw_joint_point(graph)
    else:
        # print("Fail!")
        pass
